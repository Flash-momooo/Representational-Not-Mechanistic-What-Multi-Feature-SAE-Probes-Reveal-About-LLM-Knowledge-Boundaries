"""NN17: strictly nested, prospective PARR confirmation analysis.

The protocol is frozen in paper/NN17_FROZEN_PARR_PROSPECTIVE_PROTOCOL.md.
This implementation deliberately does not read an earlier result JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_frozen_representation_controls import canonicalize, load_condition, raw_and_confidence
from scripts.poc_fpe7_observability_utility import event_onsets, group_fold_ids
from scripts.poc_fpe8_sparse_risk_distillation import fit_importance, fit_sparse_student, predict_sparse_student
from scripts.poc_nn1_sparse_compression_controls import EPS, N_FOLDS, STAGES, fit_common, safe_metrics, trajectory_max


K_VALUES = (1, 2, 4, 8, 16, 32, 64)
CANDIDATES = ("confidence", "adaptive_raw", "dense_raw")
COMPLEXITY = {"confidence": 0, "adaptive_raw": 1, "dense_raw": 2}


def sparse_probability(raw, confidence, labels, train, k, device):
    if not train.any():
        return np.full(len(labels), np.nan, dtype=np.float64)
    if len(np.unique(labels[train])) < 2:
        return np.full(len(labels), float(np.mean(labels[train])), dtype=np.float64)
    importance, _ = fit_importance(raw, confidence, labels.astype(np.float64), train, device)
    support = np.sort(np.argsort(-importance)[: min(k, raw.shape[1])]).astype(np.int32)
    model = fit_sparse_student(raw, confidence, labels, labels, train, support)
    return predict_sparse_student(model, raw, confidence)


def candidate_probabilities(raw, confidence, labels, train, k, device):
    return {
        "confidence": fit_common(np.zeros((len(labels), 0), dtype=np.float32), confidence, labels, train),
        "adaptive_raw": sparse_probability(raw, confidence, labels, train, k, device),
        "dense_raw": fit_common(raw, confidence, labels, train),
    }


def grouped_nll(labels, probability, question_ids, rows):
    values = []
    for question_id in np.unique(question_ids[rows]):
        selected = rows & (question_ids == question_id) & np.isfinite(probability)
        if not selected.any():
            continue
        p = np.clip(probability[selected], EPS, 1 - EPS)
        y = labels[selected]
        values.append(float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))))
    if not values:
        return {"mean_group_nll": math.inf, "n_questions": 0}
    return {"mean_group_nll": float(np.mean(values)), "n_questions": len(values)}


def select_k(states, table, labels, fold_ids, question_ids, fold, device):
    selection_fold = (fold + 1) % N_FOLDS
    inner_fit_fold_mask = (fold_ids != fold) & (fold_ids != selection_fold)
    selection_fold_mask = fold_ids == selection_fold
    records = []
    for k in K_VALUES:
        pooled_losses = []
        n_questions = 0
        for stage in STAGES:
            raw, confidence = raw_and_confidence(states, stage, "raw")
            risk = table["risk"][stage]
            labels_stage = table["event"][stage].astype(np.int8)
            fit = inner_fit_fold_mask & risk
            selection = selection_fold_mask & risk
            probability = sparse_probability(raw, confidence, labels_stage, fit, k, device)
            result = grouped_nll(labels_stage, probability, question_ids, selection)
            if np.isfinite(result["mean_group_nll"]):
                pooled_losses.append(result["mean_group_nll"])
                n_questions += result["n_questions"]
        records.append({"k": k, "pooled_stage_mean_group_nll": float(np.mean(pooled_losses)), "n_selection_question_stages": n_questions})
    chosen = min(records, key=lambda row: (row["pooled_stage_mean_group_nll"], row["k"]))["k"]
    return int(chosen), records


def alert_results(scores, table, labels, onsets, fold_ids):
    alert_stage = np.full(len(labels), -1, dtype=np.int8)
    thresholds = []
    for fold in range(N_FOLDS):
        test = fold_ids == fold
        calibration = fold_ids == ((fold + 2) % N_FOLDS)
        calibration_correct = calibration & (labels == 0)
        maxima = trajectory_max(scores, table)[calibration_correct]
        maxima = maxima[np.isfinite(maxima)]
        threshold = float(np.quantile(maxima, 0.90, method="higher")) if len(maxima) else 1.0
        thresholds.append({"test_fold": fold, "calibration_fold": (fold + 2) % N_FOLDS, "threshold": threshold, "n_calibration_correct": int(len(maxima))})
        for stage in STAGES:
            selected = test & table["risk"][stage] & (alert_stage < 0) & (scores[stage] >= threshold)
            alert_stage[selected] = stage
    correct = labels == 0
    eligible_error = (labels == 1) & (onsets >= STAGES[0])
    false_alert = correct & (alert_stage >= 0)
    detected = eligible_error & (alert_stage >= 0) & (alert_stage <= onsets)
    totals = {
        "n_correct": int(correct.sum()), "n_eligible_error": int(eligible_error.sum()),
        "false_alert": int(false_alert.sum()), "detected": int(detected.sum()),
        "actual_fpr": float(false_alert.sum() / max(correct.sum(), 1)),
        "pre_event_recall": float(detected.sum() / max(eligible_error.sum(), 1)),
        "mean_token_lead": float(np.mean(onsets[detected] - alert_stage[detected] + 1)) if detected.any() else 0.0,
    }
    return alert_stage, thresholds, totals


def bootstrap_utility_delta(labels, onsets, question_ids, candidate_alert, reference_alert, repeats, seed):
    groups = np.unique(question_ids)
    index = {group: np.flatnonzero(question_ids == group) for group in groups}
    rng = np.random.default_rng(seed)
    recall_deltas, fpr_deltas = [], []
    for _ in range(repeats):
        rows = np.concatenate([index[group] for group in rng.choice(groups, size=len(groups), replace=True)])
        correct = labels[rows] == 0
        eligible = (labels[rows] == 1) & (onsets[rows] >= STAGES[0])
        def summarize(alert):
            a = alert[rows]
            fpr = np.sum(correct & (a >= 0)) / max(np.sum(correct), 1)
            recall = np.sum(eligible & (a >= 0) & (a <= onsets[rows])) / max(np.sum(eligible), 1)
            return float(recall), float(fpr)
        c_recall, c_fpr = summarize(candidate_alert)
        r_recall, r_fpr = summarize(reference_alert)
        recall_deltas.append(c_recall - r_recall)
        fpr_deltas.append(c_fpr - r_fpr)
    def interval(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975))}
    return {"pre_event_recall_delta": interval(recall_deltas), "actual_fpr_delta": interval(fpr_deltas)}


def analyze(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    states, table = load_condition(tokenizer, ROOT / args.data, ROOT / args.cache)
    states = canonicalize(states)
    labels_final = states["labels"].astype(np.int8)
    question_ids = states["question_ids"].astype(str)
    fold_ids = group_fold_ids(states["pair_ids"].astype(str), seed=args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    methods = (*CANDIDATES, "parr", "ss_parr") if args.stage_structured else (*CANDIDATES, "parr")
    predictions = {name: {stage: np.full(len(labels_final), np.nan) for stage in STAGES} for name in methods}
    capacity_selection, routing = [], []
    for fold in range(N_FOLDS):
        test = fold_ids == fold
        outer_train = ~test
        selection_fold = (fold + 1) % N_FOLDS
        selection = (fold_ids == selection_fold)
        inner_fit = outer_train & ~selection
        k, k_records = select_k(states, table, labels_final, fold_ids, question_ids, fold, device)
        capacity_selection.append({"fold": fold, "selection_fold": selection_fold, "selected_k": k, "records": k_records})
        for stage in STAGES:
            raw, confidence = raw_and_confidence(states, stage, "raw")
            risk = table["risk"][stage]
            labels = table["event"][stage].astype(np.int8)
            full = candidate_probabilities(raw, confidence, labels, outer_train & risk, k, device)
            inner = candidate_probabilities(raw, confidence, labels, inner_fit & risk, k, device)
            records = []
            for method in CANDIDATES:
                record = {"method": method, **grouped_nll(labels, inner[method], question_ids, selection & risk)}
                records.append(record)
                predictions[method][stage][test & risk] = full[method][test & risk]
            chosen = min(records, key=lambda row: (row["mean_group_nll"], COMPLEXITY[row["method"]]))["method"]
            predictions["parr"][stage][test & risk] = full[chosen][test & risk]
            if args.stage_structured:
                structured_method = "adaptive_raw" if stage == 1 else "dense_raw"
                predictions["ss_parr"][stage][test & risk] = full[structured_method][test & risk]
            routing.append({"fold": fold, "stage": f"T{stage}", "selection_fold": selection_fold, "selected_k": k, "chosen": chosen, "candidates": records})
    trajectories = {method: trajectory_max(by_stage, table) for method, by_stage in predictions.items()}
    summaries = [{"method": method, **safe_metrics(labels_final, score)} for method, score in trajectories.items()]
    onsets = event_onsets(table)
    alerts, utilities, thresholds = {}, {}, {}
    for method in predictions:
        alert, threshold_rows, utility = alert_results(predictions[method], table, labels_final, onsets, fold_ids)
        alerts[method], thresholds[method], utilities[method] = alert, threshold_rows, utility
    primary_method = "ss_parr" if args.stage_structured else "parr"
    paired = {
        reference: bootstrap_utility_delta(labels_final, onsets, question_ids, alerts[primary_method], alerts[reference], args.bootstrap, args.seed + i)
        for i, reference in enumerate(CANDIDATES)
    }
    primary = paired["adaptive_raw"]
    calibration_counts = [row["n_calibration_correct"] for row in thresholds[primary_method]]
    calibration_adequate = bool(min(calibration_counts, default=0) >= args.min_calibration_successes)
    primary_pass = bool(calibration_adequate and primary["pre_event_recall_delta"]["ci95_low"] > 0 and utilities[primary_method]["actual_fpr"] - utilities["adaptive_raw"]["actual_fpr"] <= 0.03)
    payload = {
        "experiment": args.experiment,
        "protocol": args.protocol,
        "analysis_status": "prospective frozen confirmation; no earlier result file read",
        "model": args.tokenizer, "data": args.data, "cache": args.cache, "device": device,
        "n_trajectories": int(len(labels_final)), "n_questions": int(len(np.unique(question_ids))),
        "accuracy": float(np.mean(labels_final == 0)), "n_correct": int(np.sum(labels_final == 0)), "n_error": int(np.sum(labels_final == 1)),
        "selected_k_by_fold": capacity_selection,
        "routing": routing,
        "routing_counts": {f"T{stage}": dict(Counter(row["chosen"] for row in routing if row["stage"] == f"T{stage}")) for stage in STAGES},
        "trajectory_summary": summaries, "pre_event_utility_10pct_nominal": utilities,
        "thresholds": thresholds, "paired_utility_bootstrap": paired,
        "stage_structured_schedule": {"T1": "adaptive_raw", "T2": "dense_raw", "T3": "dense_raw"} if args.stage_structured else None,
        "calibration_adequacy": {"minimum_successes_per_fold": args.min_calibration_successes, "observed": calibration_counts, "passed": calibration_adequate},
        "primary_endpoint": {"method": primary_method, "reference": "adaptive_raw", "criterion": "calibration adequate; recall CI low > 0; point FPR delta <= .03", "passed": primary_pass, **primary},
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(ROOT / args.output.replace(".json", ".npz"), labels=labels_final, question_ids=question_ids, fold_ids=fold_ids, onsets=onsets, **{f"score_{method}_T{stage}": predictions[method][stage] for method in predictions for stage in STAGES}, **{f"alert_{method}": alerts[method] for method in alerts})
    print(f"saved -> {output}")
    for row in summaries:
        print(f"{row['method']:<14} AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} Brier={row['brier']:.4f}")
    print("primary", json.dumps(payload["primary_endpoint"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--experiment", default="NN17 frozen prospective PARR confirmation")
    parser.add_argument("--protocol", default="paper/NN17_FROZEN_PARR_PROSPECTIVE_PROTOCOL.md")
    parser.add_argument("--stage-structured", action="store_true")
    parser.add_argument("--min-calibration-successes", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()


