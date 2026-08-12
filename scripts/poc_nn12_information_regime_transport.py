"""NN12: grouped transport of sparse supports across information regimes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_frozen_representation_controls import (  # noqa: E402
    canonicalize,
    load_condition,
    raw_and_confidence,
)
from scripts.poc_fpe7_observability_utility import group_fold_ids  # noqa: E402
from scripts.poc_fpe8_sparse_risk_distillation import (  # noqa: E402
    fit_importance,
    fit_sparse_student,
    jaccard,
    predict_sparse_student,
)
from scripts.poc_nn1_sparse_compression_controls import (  # noqa: E402
    N_FOLDS,
    STAGES,
    grouped_bootstrap_delta,
    safe_metrics,
    trajectory_max,
)


K_VALUES = (8, 32, 64)


def run_direction(
    source_states: dict[str, np.ndarray],
    source_table: dict[str, np.ndarray],
    target_states: dict[str, np.ndarray],
    target_table: dict[str, np.ndarray],
    fold_ids: np.ndarray,
    device: str,
    seed: int,
    bootstrap: int,
) -> dict:
    labels = target_states["labels"].astype(np.int8)
    question_ids = target_states["question_ids"].astype(str)
    predictions: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    names = ["dense_target"]
    for k in K_VALUES:
        names.extend((f"target_support_k{k}", f"transported_support_k{k}"))
    for name in names:
        for stage in STAGES:
            predictions[name][stage] = np.full(len(labels), np.nan, dtype=np.float64)
    support_records = []

    for fold in range(N_FOLDS):
        for stage in STAGES:
            target_risk = target_table["risk"][stage]
            source_risk = source_table["risk"][stage]
            target_train = (fold_ids != fold) & target_risk
            target_test = (fold_ids == fold) & target_risk
            source_train = (fold_ids != fold) & source_risk
            if not target_train.any() or not target_test.any() or not source_train.any():
                continue
            target_y = target_table["event"][stage].astype(np.int8)
            source_y = source_table["event"][stage].astype(np.int8)
            target_raw, target_conf = raw_and_confidence(target_states, stage, "raw")
            source_raw, source_conf = raw_and_confidence(source_states, stage, "raw")

            dense_support = np.arange(target_raw.shape[1], dtype=np.int32)
            dense_model = fit_sparse_student(
                target_raw, target_conf, target_y, target_y, target_train, dense_support
            )
            predictions["dense_target"][stage][target_test] = predict_sparse_student(
                dense_model, target_raw, target_conf
            )[target_test]

            source_importance, _ = fit_importance(
                source_raw, source_conf, source_y.astype(np.float64), source_train, device
            )
            target_importance, _ = fit_importance(
                target_raw, target_conf, target_y.astype(np.float64), target_train, device
            )
            source_order = np.argsort(-source_importance).astype(np.int32)
            target_order = np.argsort(-target_importance).astype(np.int32)
            for k in K_VALUES:
                transported = np.sort(source_order[:k])
                local = np.sort(target_order[:k])
                for name, support in (
                    (f"target_support_k{k}", local),
                    (f"transported_support_k{k}", transported),
                ):
                    model = fit_sparse_student(
                        target_raw, target_conf, target_y, target_y, target_train, support
                    )
                    predictions[name][stage][target_test] = predict_sparse_student(
                        model, target_raw, target_conf
                    )[target_test]
                support_records.append({
                    "fold": fold,
                    "stage": f"T{stage}",
                    "k": k,
                    "jaccard": jaccard(local, transported),
                })

    scores = {name: trajectory_max(predictions[name], target_table) for name in names}
    summary = [{"method": name, **safe_metrics(labels, scores[name])} for name in names]
    comparisons = []
    for k in K_VALUES:
        candidate = f"target_support_k{k}"
        reference = f"transported_support_k{k}"
        for metric in ("auroc", "brier"):
            comparisons.append({
                "candidate": candidate,
                "reference": reference,
                **grouped_bootstrap_delta(
                    labels, scores[candidate], scores[reference], question_ids,
                    metric, repeats=bootstrap, seed=seed + len(comparisons),
                ),
            })
    stability = []
    for k in K_VALUES:
        local = [row["jaccard"] for row in support_records if row["k"] == k]
        stability.append({
            "k": k,
            "mean_jaccard": float(np.mean(local)),
            "min_jaccard": float(np.min(local)),
            "max_jaccard": float(np.max(local)),
            "n_fold_stages": len(local),
        })
    return {
        "trajectory_summary": summary,
        "paired_bootstrap": comparisons,
        "support_transport": stability,
        "support_records": support_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-only-data", required=True)
    parser.add_argument("--question-only-cache", required=True)
    parser.add_argument("--evidence-data", required=True)
    parser.add_argument("--evidence-cache", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    question_states, question_table = load_condition(
        tokenizer, ROOT / args.question_only_data, ROOT / args.question_only_cache
    )
    evidence_states, evidence_table = load_condition(
        tokenizer, ROOT / args.evidence_data, ROOT / args.evidence_cache
    )
    question_states = canonicalize(question_states)
    evidence_states = canonicalize(evidence_states)
    for key in ("question_ids", "pair_ids"):
        if not np.array_equal(question_states[key].astype(str), evidence_states[key].astype(str)):
            raise RuntimeError(f"Regime row alignment failed for {key}")
    fold_ids = group_fold_ids(question_states["pair_ids"].astype(str), seed=args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    payload = {
        "experiment": "NN12 information-regime support transport",
        "analysis_status": "post-hoc diagnostic per NN12 frozen protocol",
        "tokenizer": args.tokenizer,
        "n_questions": int(len(np.unique(question_states["question_ids"].astype(str)))),
        "n_trajectories": int(len(question_states["labels"])),
        "question_only_to_evidence": run_direction(
            question_states, question_table, evidence_states, evidence_table,
            fold_ids, device, args.seed, args.bootstrap,
        ),
        "evidence_to_question_only": run_direction(
            evidence_states, evidence_table, question_states, question_table,
            fold_ids, device, args.seed + 1000, args.bootstrap,
        ),
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output}")
    for direction in ("question_only_to_evidence", "evidence_to_question_only"):
        print(direction)
        for row in payload[direction]["trajectory_summary"]:
            print(f"  {row['method']:<26} AUROC={row['auroc']:.4f} Brier={row['brier']:.4f}")
        print("  Jaccard", payload[direction]["support_transport"])


if __name__ == "__main__":
    main()
