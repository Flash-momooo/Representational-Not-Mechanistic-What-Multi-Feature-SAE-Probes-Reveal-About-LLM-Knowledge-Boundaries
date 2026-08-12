"""FPE10 frozen compression controls and confirmatory target evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import betainc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe3_trajectory_dynamics import score_metrics
from scripts.poc_fpe7_observability_utility import event_onsets, group_fold_ids
from scripts.poc_fpe8_sparse_risk_distillation import (
    fit_importance,
    fit_sparse_student,
    predict_sparse_student,
    top_k,
)
from scripts.poc_fpe9_cross_domain_risk_calibration import threshold_metrics


STAGES = (1, 2, 3)
K = 8
TARGET_FPR = 0.10
SUCCESS_BUDGETS = (16, 32)
N_CALIBRATION_SEEDS = 200
N_RANDOM_SUPPORTS = 20
EPS = 1e-7


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prefix_status_aliases(text: str, aliases: list[str], eos: bool = False) -> str:
    first_line = None
    line_terminated = False
    for part in text.splitlines(keepends=True):
        content = part.rstrip("\r\n").strip()
        if content:
            first_line = content
            line_terminated = part.endswith(("\n", "\r"))
            break
    candidate = "" if first_line is None else re.sub(
        r"^answer\s*:\s*", "", first_line, flags=re.IGNORECASE
    ).strip()
    candidate = normalize_answer(candidate)
    gold = [normalize_answer(value) for value in aliases if normalize_answer(value)]
    if candidate and not any(value.startswith(candidate) for value in gold):
        return "event"
    if line_terminated or eos:
        return "success" if candidate in gold else "event"
    return "risk"


def load_condition(tokenizer, data_path: Path, cache_path: Path):
    rows = read_jsonl(data_path)
    archive = np.load(cache_path, allow_pickle=True)
    states = {key: archive[key] for key in archive.files}
    n = len(rows)
    risk = np.zeros((4, n), dtype=bool)
    event = np.zeros((4, n), dtype=np.int8)
    success = np.zeros((4, n), dtype=np.int8)
    transition = np.zeros((4, n), dtype=bool)
    for index, row in enumerate(rows):
        tokens = list(map(int, row["generated_token_ids"]))
        aliases = row.get("gold_answers") or [row["gold_answer"]]
        for stage in range(4):
            if stage > len(tokens):
                continue
            before = prefix_status_aliases(
                tokenizer.decode(tokens[:stage], skip_special_tokens=True), aliases
            )
            if before != "risk":
                continue
            risk[stage, index] = True
            if stage < len(tokens):
                after = prefix_status_aliases(
                    tokenizer.decode(tokens[:stage + 1], skip_special_tokens=True), aliases
                )
                transition[stage, index] = True
            else:
                after = prefix_status_aliases(
                    tokenizer.decode(tokens, skip_special_tokens=True), aliases, eos=True
                )
            event[stage, index] = int(after == "event")
            success[stage, index] = int(after == "success")
    return canonicalize(states), {
        "risk": risk, "event": event, "success": success,
        "token_transition": transition,
    }


def canonicalize(states: dict) -> dict:
    output = dict(states)
    for stage in range(4):
        if f"raw_T{stage}" not in output:
            output[f"raw_T{stage}"] = output[f"raw_T{stage}_L18"]
        if f"sae_T{stage}" not in output and f"sae_T{stage}_L18" in output:
            output[f"sae_T{stage}"] = output[f"sae_T{stage}_L18"]
    return output


def concatenate(conditions):
    state_keys = ["labels", "question_ids", "pair_ids"]
    for stage in range(4):
        state_keys.extend((
            f"valid_T{stage}", f"confidence_T{stage}", f"token_prefix_T{stage}",
            f"raw_T{stage}", f"sae_T{stage}",
        ))
    states = {}
    for key in state_keys:
        if not all(key in local for _, local, _ in conditions):
            continue
        blocks = []
        for name, local, _ in conditions:
            values = local[key]
            if key in ("question_ids", "pair_ids"):
                values = np.asarray([f"{name}::{value}" for value in values.astype(str)], dtype=object)
            blocks.append(values)
        states[key] = np.concatenate(blocks)
    table = {
        key: np.concatenate([table[key] for _, _, table in conditions], axis=1)
        for key in ("risk", "event", "success", "token_transition")
    }
    return states, table


def raw_and_confidence(states, stage: int, representation: str = "raw"):
    values = states[f"{representation}_T{stage}"].astype(np.float32)
    values -= states[f"{representation}_T0"].astype(np.float32)
    confidence = np.concatenate((
        states["confidence_T0"].astype(np.float32),
        states[f"confidence_T{stage}"].astype(np.float32),
    ), axis=1)
    return values, confidence


def standardize_fit(source: np.ndarray, train: np.ndarray):
    mean = source[train].mean(axis=0)
    scale = source[train].std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def fit_predict_common(source_x, target_x, source_conf, target_conf, labels, train):
    support = np.arange(source_x.shape[1], dtype=np.int32)
    model = fit_sparse_student(source_x, source_conf, labels, labels, train, support)
    return (
        predict_sparse_student(model, source_x, source_conf),
        predict_sparse_student(model, target_x, target_conf),
    )


def l1_support(raw, confidence, labels, train):
    mean, scale = standardize_fit(raw, train)
    x = (raw - mean) / scale
    conf_mean, conf_scale = standardize_fit(confidence, train)
    c = (confidence - conf_mean) / conf_scale
    features = np.concatenate((x, c), axis=1)
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.05, max_iter=3000,
    )
    model.fit(features[train], labels[train])
    weights = np.abs(model.coef_[0, :raw.shape[1]])
    return top_k(weights, K), int(np.count_nonzero(weights))


def trajectory_max(scores, table):
    maximum = np.full(table["risk"].shape[1], -np.inf, dtype=np.float64)
    for stage in STAGES:
        valid = table["risk"][stage]
        maximum[valid] = np.maximum(maximum[valid], scores[stage][valid])
    return maximum


def beta_quantile(values, quantile):
    values = np.sort(values[np.isfinite(values)])
    if not len(values):
        return math.nan
    a, b = (len(values) + 1) * quantile, (len(values) + 1) * (1 - quantile)
    edges = np.arange(len(values) + 1) / len(values)
    return float(np.sum(np.diff(betainc(a, b, edges)) * values))


def calibration_utility(scores, table, labels, pair_ids, onsets):
    maximum = trajectory_max(scores, table)
    unique_pairs = np.unique(pair_ids)
    rows = []
    support_failures = {}
    for seed in range(N_CALIBRATION_SEEDS):
        order = unique_pairs.copy()
        np.random.default_rng(20260716 + seed).shuffle(order)
        for budget in SUCCESS_BUDGETS:
            if int(np.sum(labels == 0)) < budget + 8:
                support_failures[budget] = (
                    f"Only {int(np.sum(labels == 0))} target-correct trajectories; "
                    f"budget {budget} plus eight held-out correct trajectories required."
                )
                continue
            calibration = np.zeros(len(labels), dtype=bool)
            correct = 0
            for pair_id in order[:-20]:
                local = pair_ids == pair_id
                calibration |= local
                correct += int(np.sum(local & (labels == 0)))
                if correct >= budget:
                    break
            if correct < budget or int(np.sum((~calibration) & (labels == 0))) < 1:
                continue
            correct_scores = maximum[calibration & (labels == 0)]
            threshold = beta_quantile(correct_scores, 1 - TARGET_FPR)
            metrics = threshold_metrics(
                threshold, scores, table["risk"], labels, onsets, ~calibration
            )
            rows.append({"seed": seed, "budget": budget, "n_calibration": int(calibration.sum()), **metrics})
    output = []
    for budget in SUCCESS_BUDGETS:
        local = [row for row in rows if row["budget"] == budget]
        if not local:
            output.append({
                "success_budget": budget,
                "supported": False,
                "reason": support_failures.get(budget, "No valid grouped calibration/test split."),
                "n_valid_splits": 0,
            })
            continue
        output.append({
            "success_budget": budget,
            "supported": True,
            "n_valid_splits": len(local),
            "mean_total_labels": float(np.mean([row["n_calibration"] for row in local])),
            "actual_fpr_mean": float(np.mean([row["actual_fpr"] for row in local])),
            "actual_fpr_p05": float(np.quantile([row["actual_fpr"] for row in local], 0.05)),
            "actual_fpr_p95": float(np.quantile([row["actual_fpr"] for row in local], 0.95)),
            "pre_event_recall_mean": float(np.mean([row["pre_event_recall"] for row in local])),
        })
    return output


def run(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.source_kind == "pooled_gemma":
        specs = (
            ("2wiki_a", ROOT / "data/v37_2wiki_trajectory.jsonl", ROOT / "outputs/cache/v37_2wiki_trajectory_states.npz"),
            ("2wiki_b", ROOT / "data/fpe5c2_policy_full_trajectory.jsonl", ROOT / "outputs/cache/fpe5c2_policy_full_trajectory_states.npz"),
        )
        conditions = []
        for name, data, cache in specs:
            states, table = load_condition(tokenizer, data, cache)
            conditions.append((name, states, table))
        source, source_table = concatenate(conditions)
    else:
        source, source_table = load_condition(
            tokenizer, ROOT / args.source_data, ROOT / args.source_cache
        )
    target, target_table = load_condition(
        tokenizer, ROOT / args.target_data, ROOT / args.target_cache
    )
    source_labels = source["labels"].astype(np.int8)
    target_labels = target["labels"].astype(np.int8)
    if args.endpoint == "early_final_outcome":
        source_table = dict(source_table)
        target_table = dict(target_table)
        source_table["risk"] = np.stack([
            source[f"valid_T{stage}"].astype(bool) for stage in range(4)
        ])
        target_table["risk"] = np.stack([
            target[f"valid_T{stage}"].astype(bool) for stage in range(4)
        ])
        source_table["event"] = np.broadcast_to(
            source_labels, source_table["risk"].shape
        ).copy()
        target_table["event"] = np.broadcast_to(
            target_labels, target_table["risk"].shape
        ).copy()
    fold_ids = group_fold_ids(source["pair_ids"].astype(str), seed=42)
    train_group = fold_ids != 0
    target_ids = target["question_ids"].astype(str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_predictions = {}
    target_predictions = {}
    supports = {}
    l1_diagnostics = {}
    source_stage_support = {}

    for stage in STAGES:
        valid = source_table["risk"][stage]
        labels = source_table["event"][stage].astype(np.int8)
        train = train_group & valid
        raw, source_conf = raw_and_confidence(source, stage, "raw")
        target_raw, target_conf = raw_and_confidence(target, stage, "raw")
        unique_labels = np.unique(labels[train])
        source_stage_support[f"T{stage}"] = {
            "n_risk": int(train.sum()),
            "n_event": int(labels[train].sum()),
            "classes": unique_labels.tolist(),
            "learnable": bool(len(unique_labels) == 2),
        }
        if len(unique_labels) < 2:
            constant = float(np.mean(labels[train])) if train.any() else 0.0
            unavailable_methods = (
                "confidence", "dense_residual", "raw_topk", "l1_raw",
                "pca8", "gaussian_rp8",
            )
            for method in unavailable_methods:
                source_predictions.setdefault(method, {})[stage] = np.full(
                    len(source_labels), constant, dtype=np.float64
                )
                target_predictions.setdefault(method, {})[stage] = np.full(
                    len(target_labels), constant, dtype=np.float64
                )
            for random_seed in range(N_RANDOM_SUPPORTS):
                method = f"random_coords_{random_seed:02d}"
                source_predictions.setdefault(method, {})[stage] = np.full(
                    len(source_labels), constant, dtype=np.float64
                )
                target_predictions.setdefault(method, {})[stage] = np.full(
                    len(target_labels), constant, dtype=np.float64
                )
            if "sae_T0" in source and "sae_T0" in target:
                source_predictions.setdefault("sae_topk", {})[stage] = np.full(
                    len(source_labels), constant, dtype=np.float64
                )
                target_predictions.setdefault("sae_topk", {})[stage] = np.full(
                    len(target_labels), constant, dtype=np.float64
                )
            continue

        empty_source = np.zeros((len(source_labels), 0), dtype=np.float32)
        empty_target = np.zeros((len(target_labels), 0), dtype=np.float32)
        source_predictions.setdefault("confidence", {})[stage], target_predictions.setdefault("confidence", {})[stage] = fit_predict_common(
            empty_source, empty_target, source_conf, target_conf, labels, train
        )
        source_predictions.setdefault("dense_residual", {})[stage], target_predictions.setdefault("dense_residual", {})[stage] = fit_predict_common(
            raw, target_raw, source_conf, target_conf, labels, train
        )

        importance, _ = fit_importance(raw, source_conf, labels.astype(float), train, device)
        raw_support = top_k(importance, K)
        supports[f"raw_topk|T{stage}"] = raw_support.tolist()
        model = fit_sparse_student(raw, source_conf, labels, labels, train, raw_support)
        source_predictions.setdefault("raw_topk", {})[stage] = predict_sparse_student(model, raw, source_conf)
        target_predictions.setdefault("raw_topk", {})[stage] = predict_sparse_student(model, target_raw, target_conf)

        support, nonzero = l1_support(raw, source_conf, labels, train)
        supports[f"l1_raw|T{stage}"] = support.tolist()
        l1_diagnostics[f"T{stage}"] = {"nonzero_before_truncation": nonzero}
        model = fit_sparse_student(raw, source_conf, labels, labels, train, support)
        source_predictions.setdefault("l1_raw", {})[stage] = predict_sparse_student(model, raw, source_conf)
        target_predictions.setdefault("l1_raw", {})[stage] = predict_sparse_student(model, target_raw, target_conf)

        mean, scale = standardize_fit(raw, train)
        source_scaled = (raw - mean) / scale
        target_scaled = (target_raw - mean) / scale
        pca = PCA(n_components=K, svd_solver="randomized", random_state=20260716)
        source_pca = np.zeros((len(raw), K), dtype=np.float32)
        source_pca[train] = pca.fit_transform(source_scaled[train]).astype(np.float32)
        other = ~train
        source_pca[other] = pca.transform(source_scaled[other]).astype(np.float32)
        target_pca = pca.transform(target_scaled).astype(np.float32)
        source_predictions.setdefault("pca8", {})[stage], target_predictions.setdefault("pca8", {})[stage] = fit_predict_common(
            source_pca, target_pca, source_conf, target_conf, labels, train
        )

        rng = np.random.default_rng(20260716 + stage)
        projection = rng.normal(size=(raw.shape[1], K)).astype(np.float32) / math.sqrt(K)
        source_rp = source_scaled @ projection
        target_rp = target_scaled @ projection
        source_predictions.setdefault("gaussian_rp8", {})[stage], target_predictions.setdefault("gaussian_rp8", {})[stage] = fit_predict_common(
            source_rp, target_rp, source_conf, target_conf, labels, train
        )

        for random_seed in range(N_RANDOM_SUPPORTS):
            rng = np.random.default_rng(20260716 + 100 * stage + random_seed)
            random_support = np.sort(rng.choice(raw.shape[1], size=K, replace=False)).astype(np.int32)
            model = fit_sparse_student(raw, source_conf, labels, labels, train, random_support)
            method = f"random_coords_{random_seed:02d}"
            source_predictions.setdefault(method, {})[stage] = predict_sparse_student(model, raw, source_conf)
            target_predictions.setdefault(method, {})[stage] = predict_sparse_student(model, target_raw, target_conf)

        if "sae_T0" in source and "sae_T0" in target:
            sae, _ = raw_and_confidence(source, stage, "sae")
            target_sae, _ = raw_and_confidence(target, stage, "sae")
            importance, _ = fit_importance(sae, source_conf, labels.astype(float), train, device)
            sae_support = top_k(importance, K)
            supports[f"sae_topk|T{stage}"] = sae_support.tolist()
            model = fit_sparse_student(sae, source_conf, labels, labels, train, sae_support)
            source_predictions.setdefault("sae_topk", {})[stage] = predict_sparse_student(model, sae, source_conf)
            target_predictions.setdefault("sae_topk", {})[stage] = predict_sparse_student(model, target_sae, target_conf)

    stage_summary = []
    trajectory_summary = []
    utility = {}
    onsets = event_onsets(target_table)
    for method, scores in target_predictions.items():
        for stage in STAGES:
            valid = target_table["risk"][stage]
            labels = target_table["event"][stage]
            probability = scores[stage]
            within, _ = score_metrics(labels, probability, target_ids, valid)
            row = {
                "method": method, "stage": f"T{stage}",
                "n": int(valid.sum()), "events": int(labels[valid].sum()),
                "auroc": float(roc_auc_score(labels[valid], probability[valid])) if len(np.unique(labels[valid])) == 2 else None,
                "auprc": float(average_precision_score(labels[valid], probability[valid])) if labels[valid].sum() else None,
                "within_question_auroc": within["within_question_auroc_macro"],
            }
            stage_summary.append(row)
        maximum = trajectory_max(scores, target_table)
        finite = np.isfinite(maximum)
        trajectory_summary.append({
            "method": method,
            "n": int(finite.sum()),
            "auroc": float(roc_auc_score(target_labels[finite], maximum[finite])) if len(np.unique(target_labels[finite])) == 2 else None,
            "auprc": float(average_precision_score(target_labels[finite], maximum[finite])) if target_labels[finite].sum() else None,
        })
        if args.endpoint == "next_token_hazard" and not method.startswith("random_coords_"):
            utility[method] = calibration_utility(
                scores, target_table, target_labels, target["pair_ids"].astype(str), onsets
            )

    random_rows = [row for row in trajectory_summary if row["method"].startswith("random_coords_")]
    random_distribution = {
        metric: {
            "mean": float(np.mean([row[metric] for row in random_rows])),
            "p05": float(np.quantile([row[metric] for row in random_rows], 0.05)),
            "p95": float(np.quantile([row[metric] for row in random_rows], 0.95)),
        }
        for metric in ("auroc", "auprc")
    }
    output = ROOT / args.output
    score_cache = ROOT / "outputs" / "cache" / f"{output.stem}_scores.npz"
    score_payload = {
        "source_labels": source_labels,
        "source_question_ids": source["question_ids"].astype(str),
        "source_pair_ids": source["pair_ids"].astype(str),
        "source_risk": source_table["risk"],
        "source_event": source_table["event"],
        "target_labels": target_labels,
        "target_question_ids": target["question_ids"].astype(str),
        "target_pair_ids": target["pair_ids"].astype(str),
        "target_risk": target_table["risk"],
        "target_event": target_table["event"],
        "target_onsets": onsets,
    }
    for method, scores in target_predictions.items():
        for stage in STAGES:
            score_payload[f"score_{method}_T{stage}"] = scores[stage]
            score_payload[f"source_score_{method}_T{stage}"] = source_predictions[method][stage]
    np.savez_compressed(score_cache, **score_payload)
    payload = {
        "experiment": "FPE10/FPE12 frozen representation controls",
        "endpoint": args.endpoint,
        "source_kind": args.source_kind,
        "target": args.target_data,
        "k": K,
        "n_source": len(source_labels),
        "n_target": len(target_labels),
        "target_correct": int(np.sum(target_labels == 0)),
        "target_error": int(np.sum(target_labels == 1)),
        "supports": supports,
        "source_stage_support": source_stage_support,
        "l1_diagnostics": l1_diagnostics,
        "stage_summary": stage_summary,
        "trajectory_summary": trajectory_summary,
        "random_coordinate_distribution": random_distribution,
        "success_budget_utility": utility,
        "score_cache": str(score_cache.relative_to(ROOT)),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output}")
    for row in trajectory_summary:
        if not row["method"].startswith("random_coords_"):
            print(f"{row['method']:<18} AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f}")
    print("random", random_distribution)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-kind", choices=("pooled_gemma", "generic"), required=True)
    parser.add_argument("--source-data")
    parser.add_argument("--source-cache")
    parser.add_argument("--target-data", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--endpoint",
        choices=("next_token_hazard", "early_final_outcome"),
        default="next_token_hazard",
    )
    args = parser.parse_args()
    if args.source_kind == "generic" and (not args.source_data or not args.source_cache):
        parser.error("generic source requires --source-data and --source-cache")
    run(args)


if __name__ == "__main__":
    main()
