"""Evaluate the isolated Qwen-7B SAE on untouched NN19 WebQuestions data."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_collect_generic import read_jsonl
from scripts.poc_fpe10_frozen_representation_controls import load_condition, raw_and_confidence
from scripts.poc_fpe7_observability_utility import event_onsets, group_fold_ids
from scripts.poc_fpe8_sparse_risk_distillation import fit_importance, fit_sparse_student, predict_sparse_student
from scripts.poc_nn1_sparse_compression_controls import EPS, N_FOLDS, STAGES, fit_common, safe_metrics, trajectory_max
from scripts.poc_nn17_frozen_parr_confirmation import alert_results, grouped_nll
from scripts.poc_nn32_train_qwen7b_true_sae import TopKSAE


DATA = ROOT / "data" / "nn19_qwen7b_webquestions_trajectory.jsonl"
BASE_CACHE = ROOT / "outputs" / "cache" / "nn19_qwen7b_webquestions_trajectory_states.npz"
SAE_CACHE = ROOT / "outputs" / "cache" / "nn32_qwen7b_webquestions_true_sae_states.npz"
WEIGHTS = ROOT / "outputs" / "nn32_qwen7b_l18_topk_sae.pt"
RESULT = ROOT / "outputs" / "poc_nn32_qwen7b_true_sae_readout.json"
TOKENIZER = "models/Qwen2.5-7B-Instruct"
K_VALUES = (8, 32)
SEED = 20260730


def encode_cache() -> dict:
    base = np.load(BASE_CACHE, allow_pickle=True)
    if SAE_CACHE.exists():
        loaded = np.load(SAE_CACHE, allow_pickle=True)
        required = {f"sae_T{stage}" for stage in range(4)}
        if required.issubset(loaded.files):
            return {key: loaded[key] for key in loaded.files}
    checkpoint = torch.load(WEIGHTS, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    sae = TopKSAE(config["d_model"], config["d_sae"], config["top_k"], checkpoint["state_dict"]["mean"]).cuda().eval()
    sae.load_state_dict(checkpoint["state_dict"])
    output = {key: base[key] for key in base.files}
    with torch.inference_mode():
        for stage in range(4):
            values = base[f"raw_T{stage}"].astype(np.float32)
            blocks = []
            for start in range(0, len(values), 256):
                latent = sae.encode(torch.as_tensor(values[start:start + 256], device="cuda"))
                blocks.append(latent.cpu().numpy().astype(np.float16))
            output[f"sae_T{stage}"] = np.concatenate(blocks, axis=0)
    SAE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SAE_CACHE, **output)
    return output


def sparse_probability(values, confidence, labels, train, k):
    if len(np.unique(labels[train])) < 2:
        return np.full(len(labels), float(np.mean(labels[train])), dtype=np.float64), np.empty(0, dtype=np.int32)
    importance, _ = fit_importance(values, confidence, labels.astype(float), train, "cuda")
    support = np.sort(np.argsort(-importance)[:min(k, values.shape[1])]).astype(np.int32)
    model = fit_sparse_student(values, confidence, labels, labels, train, support)
    return predict_sparse_student(model, values, confidence), support


def choose_k(states, table, representation, fold_ids, question_ids, outer_fold):
    selection_fold = (outer_fold + 1) % N_FOLDS
    fit = (fold_ids != outer_fold) & (fold_ids != selection_fold)
    selection = fold_ids == selection_fold
    records = []
    for k in K_VALUES:
        losses = []
        for stage in STAGES:
            values, confidence = raw_and_confidence(states, stage, representation)
            risk = table["risk"][stage]
            labels = table["event"][stage].astype(np.int8)
            probability, _ = sparse_probability(values, confidence, labels, fit & risk, k)
            row = grouped_nll(labels, probability, question_ids, selection & risk)
            if np.isfinite(row["mean_group_nll"]):
                losses.append(row["mean_group_nll"])
        records.append({"k": k, "mean_group_nll": float(np.mean(losses)) if losses else float("inf")})
    return min(records, key=lambda row: (row["mean_group_nll"], row["k"]))["k"], records


def jaccard(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 1.0


def main() -> None:
    states = encode_cache()
    # load_condition creates the frozen pre-event masks from the untouched rows.
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    states, table = load_condition(tokenizer, DATA, SAE_CACHE)
    labels_final = states["labels"].astype(np.int8)
    question_ids = states["question_ids"].astype(str)
    fold_ids = group_fold_ids(states["pair_ids"].astype(str), seed=SEED)
    methods = ("confidence", "dense_raw", "raw_topk", "sae_topk")
    predictions = {method: {stage: np.full(len(labels_final), np.nan) for stage in STAGES} for method in methods}
    selections, supports = [], {"raw_topk": {}, "sae_topk": {}}
    for fold in range(N_FOLDS):
        outer_train, test = fold_ids != fold, fold_ids == fold
        raw_k, raw_records = choose_k(states, table, "raw", fold_ids, question_ids, fold)
        sae_k, sae_records = choose_k(states, table, "sae", fold_ids, question_ids, fold)
        selections.append({"fold": fold, "selection_fold": (fold + 1) % N_FOLDS, "raw_k": raw_k, "raw_candidates": raw_records, "sae_k": sae_k, "sae_candidates": sae_records})
        for stage in STAGES:
            risk = table["risk"][stage]
            labels = table["event"][stage].astype(np.int8)
            raw, confidence = raw_and_confidence(states, stage, "raw")
            sae, _ = raw_and_confidence(states, stage, "sae")
            predictions["confidence"][stage] = np.where(risk & test, fit_common(np.zeros((len(labels), 0), dtype=np.float32), confidence, labels, outer_train & risk), predictions["confidence"][stage])
            predictions["dense_raw"][stage] = np.where(risk & test, fit_common(raw, confidence, labels, outer_train & risk), predictions["dense_raw"][stage])
            raw_probability, raw_support = sparse_probability(raw, confidence, labels, outer_train & risk, raw_k)
            sae_probability, sae_support = sparse_probability(sae, confidence, labels, outer_train & risk, sae_k)
            predictions["raw_topk"][stage][risk & test] = raw_probability[risk & test]
            predictions["sae_topk"][stage][risk & test] = sae_probability[risk & test]
            supports["raw_topk"][f"fold{fold}_T{stage}"] = raw_support.tolist()
            supports["sae_topk"][f"fold{fold}_T{stage}"] = sae_support.tolist()
    trajectories = {method: trajectory_max(by_stage, table) for method, by_stage in predictions.items()}
    summaries = [{"method": method, **safe_metrics(labels_final, score)} for method, score in trajectories.items()]
    onsets = event_onsets(table)
    utilities, thresholds = {}, {}
    for method in methods:
        _, threshold, utility = alert_results(predictions[method], table, labels_final, onsets, fold_ids)
        utilities[method], thresholds[method] = utility, threshold
    stability = {}
    for family, values in supports.items():
        stage_rows = {}
        for stage in STAGES:
            sets = [values[f"fold{fold}_T{stage}"] for fold in range(N_FOLDS)]
            stage_rows[f"T{stage}"] = float(np.mean([jaccard(a, b) for a, b in combinations(sets, 2)]))
        stability[family] = stage_rows
    payload = {
        "experiment": "NN32 frozen true-SAE Qwen2.5-7B readout replication",
        "protocol": "paper/NN32_FROZEN_QWEN7B_TRUE_SAE_PROTOCOL.md",
        "analysis_status": "WebQuestions labels and folds are used only after independently trained SAE weights are fixed.",
        "model": "Qwen2.5-7B-Instruct NF4 double-quant, bfloat16 compute",
        "sae_weights": str(WEIGHTS.relative_to(ROOT)),
        "data": str(DATA.relative_to(ROOT)),
        "n_questions": int(len(np.unique(question_ids))), "n_trajectories": int(len(labels_final)),
        "accuracy": float(np.mean(labels_final == 0)),
        "selections": selections, "trajectory_summary": summaries,
        "pre_event_utility_10pct_nominal": utilities, "thresholds": thresholds,
        "support_jaccard_across_outer_folds": stability,
        "limits": "The SAE was trained on only 192 prompt-level training examples and is a small reconstruction-validated dictionary, not a public large-scale SAE suite. Nominal FPR is not a calibrated safety guarantee.",
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {RESULT}")
    for row in summaries:
        print(f"{row['method']}: AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} Brier={row['brier']:.4f}")
    print(json.dumps(utilities, indent=2))


if __name__ == "__main__":
    main()
