"""NN1: leakage-safe sparse compression controls for dynamic risk readouts.

Every reported prediction is out-of-fold by question group. Standardization,
supervised coordinate selection, L1 selection, PCA, and model fitting are
repeated inside each outer training fold. This script is an analysis of frozen
trajectory caches; it does not regenerate or relabel model outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_frozen_representation_controls import (  # noqa: E402
    canonicalize,
    load_condition,
    raw_and_confidence,
    standardize_fit,
)
from scripts.poc_fpe3_trajectory_dynamics import score_metrics  # noqa: E402
from scripts.poc_fpe7_observability_utility import (  # noqa: E402
    event_onsets,
    group_fold_ids,
)
from scripts.poc_fpe8_sparse_risk_distillation import (  # noqa: E402
    fit_importance,
    fit_sparse_student,
    jaccard,
    predict_sparse_student,
    top_k,
    utility_summary,
)


STAGES = (1, 2, 3)
DEFAULT_K = (1, 2, 4, 8, 16, 32, 64)
N_FOLDS = 5
N_RANDOM_K8 = 20
N_RANDOM_OTHER = 5
EPS = 1e-7


def parse_k_values(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not values or min(values) < 1:
        raise ValueError("K values must be positive integers")
    return values


def safe_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    valid = np.isfinite(probabilities)
    labels = labels[valid].astype(np.int8)
    probabilities = np.clip(probabilities[valid], EPS, 1 - EPS)
    if not len(labels):
        return {"n": 0, "events": 0, "auroc": None, "auprc": None,
                "brier": None, "nll_nats": None}
    two_classes = len(np.unique(labels)) == 2
    return {
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, probabilities)) if two_classes else None,
        "auprc": float(average_precision_score(labels, probabilities)) if labels.sum() else None,
        "brier": float(brier_score_loss(labels, probabilities)),
        "nll_nats": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def fit_common(
    features: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    if len(np.unique(labels[train])) < 2:
        return np.full(len(labels), float(np.mean(labels[train])), dtype=np.float64)
    support = np.arange(features.shape[1], dtype=np.int32)
    model = fit_sparse_student(features, confidence, labels, labels, train, support)
    return predict_sparse_student(model, features, confidence)


def l1_order(
    raw: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, int]:
    mean, scale = standardize_fit(raw, train)
    conf_mean, conf_scale = standardize_fit(confidence, train)
    features = np.concatenate(((raw - mean) / scale, (confidence - conf_mean) / conf_scale), axis=1)
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.05, max_iter=3000, random_state=20260726
    )
    model.fit(features[train], labels[train])
    weights = np.abs(model.coef_[0, : raw.shape[1]])
    order = np.argsort(-weights).astype(np.int32)
    return order, int(np.count_nonzero(weights))


def select_capacity_one_se(
    representation: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    question_ids: np.ndarray,
    k_values: tuple[int, ...],
    device: str,
) -> tuple[int, list[dict]]:
    """Select the smallest K statistically tied with validation-best NLL."""
    if not train.any() or not validation.any() or len(np.unique(labels[train])) < 2:
        return min(k_values), []
    importance, _ = fit_importance(
        representation, confidence, labels.astype(np.float64), train, device
    )
    order = np.argsort(-importance).astype(np.int32)
    unique_questions = np.unique(question_ids[validation])
    records = []
    for k in k_values:
        support = np.sort(order[: min(k, representation.shape[1])])
        model = fit_sparse_student(
            representation, confidence, labels, labels, train, support
        )
        probability = np.clip(
            predict_sparse_student(model, representation, confidence), EPS, 1 - EPS
        )
        losses = -(
            labels * np.log(probability) + (1 - labels) * np.log(1 - probability)
        )
        grouped_losses = np.asarray([
            float(np.mean(losses[validation & (question_ids == question_id)]))
            for question_id in unique_questions
        ])
        records.append({
            "k": int(k),
            "mean_group_nll": float(grouped_losses.mean()),
            "se_group_nll": float(grouped_losses.std(ddof=1) / math.sqrt(len(grouped_losses)))
            if len(grouped_losses) > 1 else 0.0,
            "n_validation_questions": int(len(grouped_losses)),
        })
    best = min(records, key=lambda row: row["mean_group_nll"])
    threshold = best["mean_group_nll"] + best["se_group_nll"]
    selected = min(row["k"] for row in records if row["mean_group_nll"] <= threshold)
    for row in records:
        row["best_mean_plus_one_se"] = float(threshold)
        row["selected"] = bool(row["k"] == selected)
    return selected, records


def select_global_capacity(
    states: dict[str, np.ndarray],
    table: dict[str, np.ndarray],
    fold_ids: np.ndarray,
    outer_fold: int,
    question_ids: np.ndarray,
    representation_name: str,
    k_values: tuple[int, ...],
    device: str,
) -> tuple[int, list[dict]]:
    """Choose one K across stages using a disjoint rotating validation fold."""
    selection_fold = (outer_fold + 1) % N_FOLDS
    losses_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    questions_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    for stage in STAGES:
        representation, confidence = raw_and_confidence(
            states, stage, representation_name
        )
        risk = table["risk"][stage]
        labels = table["event"][stage].astype(np.int8)
        train = (fold_ids != outer_fold) & (fold_ids != selection_fold) & risk
        validation = (fold_ids == selection_fold) & risk
        if not train.any() or not validation.any() or len(np.unique(labels[train])) < 2:
            continue
        importance, _ = fit_importance(
            representation, confidence, labels.astype(np.float64), train, device
        )
        order = np.argsort(-importance).astype(np.int32)
        for k in k_values:
            support = np.sort(order[: min(k, representation.shape[1])])
            model = fit_sparse_student(
                representation, confidence, labels, labels, train, support
            )
            probability = np.clip(
                predict_sparse_student(model, representation, confidence), EPS, 1 - EPS
            )
            local_y = labels[validation]
            local_p = probability[validation]
            losses_by_k[k].append(-(
                local_y * np.log(local_p) + (1 - local_y) * np.log(1 - local_p)
            ))
            questions_by_k[k].append(question_ids[validation])
    records = []
    for k in k_values:
        if not losses_by_k[k]:
            continue
        losses = np.concatenate(losses_by_k[k])
        questions = np.concatenate(questions_by_k[k]).astype(str)
        grouped = np.asarray([
            float(np.mean(losses[questions == question_id]))
            for question_id in np.unique(questions)
        ])
        records.append({
            "k": int(k),
            "mean_group_nll": float(grouped.mean()),
            "se_group_nll": float(grouped.std(ddof=1) / math.sqrt(len(grouped)))
            if len(grouped) > 1 else 0.0,
            "n_validation_questions": int(len(grouped)),
        })
    if not records:
        return min(k_values), []
    best = min(records, key=lambda row: row["mean_group_nll"])
    for row in records:
        row["selected"] = bool(row["k"] == best["k"])
    return int(best["k"]), records


def trajectory_max(scores: dict[int, np.ndarray], table: dict[str, np.ndarray]) -> np.ndarray:
    output = np.full(table["risk"].shape[1], -np.inf, dtype=np.float64)
    for stage in STAGES:
        valid = table["risk"][stage] & np.isfinite(scores[stage])
        output[valid] = np.maximum(output[valid], scores[stage][valid])
    output[~np.isfinite(output)] = np.nan
    return output


def grouped_bootstrap_delta(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    metric: str,
    repeats: int,
    seed: int,
) -> dict:
    unique = np.unique(groups.astype(str))
    group_rows = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        valid = np.isfinite(candidate[rows]) & np.isfinite(reference[rows])
        local_y = labels[rows][valid]
        if len(np.unique(local_y)) < 2:
            continue
        local_c = candidate[rows][valid]
        local_r = reference[rows][valid]
        if metric == "auroc":
            delta = roc_auc_score(local_y, local_c) - roc_auc_score(local_y, local_r)
        elif metric == "auprc":
            delta = average_precision_score(local_y, local_c) - average_precision_score(local_y, local_r)
        elif metric == "brier":
            delta = brier_score_loss(local_y, local_c) - brier_score_loss(local_y, local_r)
        else:
            raise ValueError(metric)
        deltas.append(float(delta))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "metric": metric,
        "orientation": "candidate-reference; lower is better only for brier",
        "mean": float(values.mean()) if len(values) else None,
        "ci95_low": float(np.quantile(values, 0.025)) if len(values) else None,
        "ci95_high": float(np.quantile(values, 0.975)) if len(values) else None,
        "probability_positive": float(np.mean(values > 0)) if len(values) else None,
        "n_valid": int(len(values)),
    }


def method_key(name: str, k: int | None = None, random_seed: int | None = None) -> str:
    if random_seed is not None:
        return f"random_coords_k{k}_s{random_seed:02d}"
    return name if k is None else f"{name}_k{k}"


def random_support_count(k: int) -> int:
    return N_RANDOM_K8 if k == 8 else N_RANDOM_OTHER


def run(args: argparse.Namespace) -> dict:
    k_values = parse_k_values(args.k_values)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    states, table = load_condition(tokenizer, ROOT / args.data, ROOT / args.cache)
    states = canonicalize(states)
    final_labels = states["labels"].astype(np.int8)
    question_ids = states["question_ids"].astype(str)
    pair_ids = states["pair_ids"].astype(str)
    fold_ids = group_fold_ids(pair_ids, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    has_sae = all(f"sae_T{stage}" in states for stage in range(4))
    dimensions = {
        "raw": int(states["raw_T0"].shape[1]),
        "sae": int(states["sae_T0"].shape[1]) if has_sae else None,
        "confidence": int(states["confidence_T0"].shape[1] * 2),
    }

    predictions: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    base_methods = ("confidence", "dense_raw", "adaptive_raw")
    for name in base_methods:
        for stage in STAGES:
            predictions[name][stage] = np.full(len(final_labels), np.nan)
    if has_sae:
        for stage in STAGES:
            predictions["adaptive_sae"][stage] = np.full(len(final_labels), np.nan)
    for k in k_values:
        for family in ("supervised_raw", "l1_raw", "pca", "gaussian_rp"):
            for stage in STAGES:
                predictions[method_key(family, k)][stage] = np.full(len(final_labels), np.nan)
        if has_sae:
            for stage in STAGES:
                predictions[method_key("supervised_sae", k)][stage] = np.full(len(final_labels), np.nan)
        for random_seed in range(random_support_count(k)):
            for stage in STAGES:
                predictions[method_key("random", k, random_seed)][stage] = np.full(len(final_labels), np.nan)

    support_store: dict[tuple[int, int, str, int], np.ndarray] = {}
    fit_diagnostics = []
    capacity_selection = []
    for fold in range(N_FOLDS):
        outer_test = fold_ids == fold
        outer_train = ~outer_test
        selected_global_raw_k, global_raw_records = select_global_capacity(
            states,
            table,
            fold_ids,
            fold,
            question_ids,
            "raw",
            k_values,
            device,
        )
        selected_global_sae_k = None
        global_sae_records = []
        if has_sae:
            selected_global_sae_k, global_sae_records = select_global_capacity(
                states,
                table,
                fold_ids,
                fold,
                question_ids,
                "sae",
                k_values,
                device,
            )
        capacity_selection.append({
            "fold": fold,
            "selection_fold": int((fold + 1) % N_FOLDS),
            "raw_selected_k": int(selected_global_raw_k),
            "raw_candidates": global_raw_records,
            "sae_selected_k": int(selected_global_sae_k)
            if selected_global_sae_k is not None else None,
            "sae_candidates": global_sae_records,
        })
        for stage in STAGES:
            risk = table["risk"][stage]
            labels = table["event"][stage].astype(np.int8)
            train = outer_train & risk
            test = outer_test & risk
            if not test.any() or not train.any():
                continue
            raw, confidence = raw_and_confidence(states, stage, "raw")
            empty = np.zeros((len(labels), 0), dtype=np.float32)
            predictions["confidence"][stage][test] = fit_common(empty, confidence, labels, train)[test]
            predictions["dense_raw"][stage][test] = fit_common(raw, confidence, labels, train)[test]
            if len(np.unique(labels[train])) < 2:
                constant = float(np.mean(labels[train]))
                for name in predictions:
                    predictions[name][stage][test] = constant
                continue

            selected_raw_k = selected_global_raw_k

            importance, importance_diag = fit_importance(
                raw, confidence, labels.astype(np.float64), train, device
            )
            raw_order = np.argsort(-importance).astype(np.int32)
            sparse_order, nonzero = l1_order(raw, confidence, labels, train)
            raw_mean, raw_scale = standardize_fit(raw, train)
            raw_scaled = (raw - raw_mean) / raw_scale
            max_k = min(max(k_values), raw.shape[1], int(train.sum()) - 1)
            pca = PCA(n_components=max_k, svd_solver="randomized", random_state=args.seed + fold * 10 + stage)
            pca_features = pca.fit(raw_scaled[train]).transform(raw_scaled).astype(np.float32)

            rng = np.random.default_rng(args.seed + fold * 100 + stage)
            projection = rng.normal(size=(raw.shape[1], max_k)).astype(np.float32) / math.sqrt(max_k)
            rp_features = raw_scaled @ projection

            sae = None
            sae_order = None
            if has_sae:
                sae, _ = raw_and_confidence(states, stage, "sae")
                sae_importance, _ = fit_importance(
                    sae, confidence, labels.astype(np.float64), train, device
                )
                sae_order = np.argsort(-sae_importance).astype(np.int32)

            selected_sae_k = selected_global_sae_k

            for k in k_values:
                local_k = min(k, raw.shape[1])
                selected = np.sort(raw_order[:local_k])
                model = fit_sparse_student(raw, confidence, labels, labels, train, selected)
                predictions[method_key("supervised_raw", k)][stage][test] = predict_sparse_student(
                    model, raw, confidence
                )[test]
                support_store[(fold, stage, "supervised_raw", k)] = selected

                selected_l1 = np.sort(sparse_order[:local_k])
                model = fit_sparse_student(raw, confidence, labels, labels, train, selected_l1)
                predictions[method_key("l1_raw", k)][stage][test] = predict_sparse_student(
                    model, raw, confidence
                )[test]

                predictions[method_key("pca", k)][stage][test] = fit_common(
                    pca_features[:, : min(k, max_k)], confidence, labels, train
                )[test]
                predictions[method_key("gaussian_rp", k)][stage][test] = fit_common(
                    rp_features[:, : min(k, max_k)], confidence, labels, train
                )[test]

                if has_sae and sae is not None and sae_order is not None:
                    sae_support = np.sort(sae_order[: min(k, sae.shape[1])])
                    model = fit_sparse_student(sae, confidence, labels, labels, train, sae_support)
                    predictions[method_key("supervised_sae", k)][stage][test] = predict_sparse_student(
                        model, sae, confidence
                    )[test]
                    support_store[(fold, stage, "supervised_sae", k)] = sae_support

                for random_seed in range(random_support_count(k)):
                    random_rng = np.random.default_rng(
                        args.seed + 10000 * random_seed + 100 * fold + stage
                    )
                    random_support = np.sort(
                        random_rng.choice(raw.shape[1], size=local_k, replace=False)
                    ).astype(np.int32)
                    model = fit_sparse_student(raw, confidence, labels, labels, train, random_support)
                    predictions[method_key("random", k, random_seed)][stage][test] = predict_sparse_student(
                        model, raw, confidence
                    )[test]

            adaptive_support = np.sort(raw_order[: min(selected_raw_k, raw.shape[1])])
            adaptive_model = fit_sparse_student(
                raw, confidence, labels, labels, train, adaptive_support
            )
            predictions["adaptive_raw"][stage][test] = predict_sparse_student(
                adaptive_model, raw, confidence
            )[test]
            if has_sae and sae is not None and sae_order is not None and selected_sae_k is not None:
                adaptive_sae_support = np.sort(
                    sae_order[: min(selected_sae_k, sae.shape[1])]
                )
                adaptive_sae_model = fit_sparse_student(
                    sae, confidence, labels, labels, train, adaptive_sae_support
                )
                predictions["adaptive_sae"][stage][test] = predict_sparse_student(
                    adaptive_sae_model, sae, confidence
                )[test]

            fit_diagnostics.append({
                "fold": fold,
                "stage": f"T{stage}",
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "events_train": int(labels[train].sum()),
                "events_test": int(labels[test].sum()),
                "l1_nonzero_before_truncation": nonzero,
                "importance": importance_diag,
            })

    stage_summary = []
    trajectory_summary = []
    trajectory_scores = {}
    for name, stage_scores in predictions.items():
        for stage in STAGES:
            risk = table["risk"][stage]
            labels = table["event"][stage].astype(np.int8)
            scores = stage_scores[stage]
            metrics = safe_metrics(labels[risk], scores[risk])
            within, _ = score_metrics(labels, scores, question_ids, risk & np.isfinite(scores))
            stage_summary.append({
                "method": name,
                "stage": f"T{stage}",
                **metrics,
                "within_question_auroc": within["within_question_auroc_macro"],
            })
        maxima = trajectory_max(stage_scores, table)
        trajectory_scores[name] = maxima
        trajectory_summary.append({"method": name, **safe_metrics(final_labels, maxima)})

    random_distribution = []
    for k in k_values:
        rows = [
            row for row in trajectory_summary
            if row["method"].startswith(f"random_coords_k{k}_")
        ]
        random_distribution.append({
            "k": k,
            "n_supports": len(rows),
            **{
                metric: {
                    "mean": float(np.mean([row[metric] for row in rows])),
                    "p05": float(np.quantile([row[metric] for row in rows], 0.05)),
                    "p95": float(np.quantile([row[metric] for row in rows], 0.95)),
                }
                for metric in ("auroc", "auprc", "brier")
            },
        })

    comparisons = []
    candidates = [
        "adaptive_raw", "supervised_raw_k8", "l1_raw_k8", "pca_k8", "gaussian_rp_k8"
    ]
    if has_sae:
        candidates.extend(("adaptive_sae", "supervised_sae_k8"))
    for candidate in candidates:
        for reference in ("dense_raw", "confidence"):
            for metric in ("auroc", "auprc", "brier"):
                comparisons.append({
                    "candidate": candidate,
                    "reference": reference,
                    **grouped_bootstrap_delta(
                        final_labels,
                        trajectory_scores[candidate],
                        trajectory_scores[reference],
                        question_ids,
                        metric,
                        repeats=args.bootstrap,
                        seed=args.seed + len(comparisons),
                    ),
                })

    support_stability = []
    for family in ("supervised_raw", "supervised_sae"):
        if family == "supervised_sae" and not has_sae:
            continue
        for k in k_values:
            fold_values = []
            stage_values = []
            for stage in STAGES:
                for left in range(N_FOLDS):
                    for right in range(left + 1, N_FOLDS):
                        fold_values.append(jaccard(
                            support_store[(left, stage, family, k)],
                            support_store[(right, stage, family, k)],
                        ))
            for fold in range(N_FOLDS):
                for left, right in zip(STAGES[:-1], STAGES[1:]):
                    stage_values.append(jaccard(
                        support_store[(fold, left, family, k)],
                        support_store[(fold, right, family, k)],
                    ))
            support_stability.append({
                "family": family,
                "k": k,
                "cross_fold_jaccard_mean": float(np.mean(fold_values)),
                "adjacent_stage_jaccard_mean": float(np.mean(stage_values)),
            })

    utility_predictions = {}
    utility_methods = []
    for name in ("confidence", "dense_raw", "adaptive_raw", "supervised_raw_k8", "pca_k8"):
        utility_predictions[(name, None, 1)] = predictions[name][1]
        utility_predictions[(name, None, 2)] = predictions[name][2]
        utility_predictions[(name, None, 3)] = predictions[name][3]
        utility_methods.append((name, None))
    if has_sae:
        for name in ("adaptive_sae", "supervised_sae_k8"):
            for stage in STAGES:
                utility_predictions[(name, None, stage)] = predictions[name][stage]
            utility_methods.append((name, None))
    utility = utility_summary(
        utility_predictions,
        table,
        final_labels,
        event_onsets(table),
        fold_ids,
        utility_methods,
        STAGES,
    )

    payload = {
        "experiment": "NN1 grouped sparse compression controls",
        "analysis_status": args.analysis_status,
        "data": args.data,
        "cache": args.cache,
        "tokenizer": args.tokenizer,
        "device": device,
        "seed": args.seed,
        "n_trajectories": int(len(final_labels)),
        "n_questions": int(len(np.unique(question_ids))),
        "has_sae": has_sae,
        "dimensions": dimensions,
        "k_values": list(k_values),
        "protocol": {
            "outer_split": "five pair-id-grouped folds",
            "leakage_control": "all transforms and supports refit inside outer training fold",
            "endpoint": "next-token first-divergence hazard at T1-T3",
            "trajectory_score": "maximum available OOF hazard probability over T1-T3",
            "random_supports": {
                "k8": N_RANDOM_K8,
                "other_k": N_RANDOM_OTHER,
            },
            "bootstrap": f"question-grouped paired bootstrap, {args.bootstrap} repeats",
        },
        "fit_diagnostics": fit_diagnostics,
        "capacity_selection": capacity_selection,
        "stage_summary": stage_summary,
        "trajectory_summary": trajectory_summary,
        "random_coordinate_distribution": random_distribution,
        "paired_bootstrap": comparisons,
        "support_stability": support_stability,
        "pre_event_utility": utility,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output}")
    for row in trajectory_summary:
        if row["method"] in (
            "confidence", "dense_raw", "adaptive_raw", "supervised_raw_k8",
            "pca_k8", "adaptive_sae", "supervised_sae_k8",
        ):
            print(
                f"{row['method']:<22} AUROC={row['auroc']:.4f} "
                f"AUPRC={row['auprc']:.4f} Brier={row['brier']:.4f}"
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K)))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--analysis-status",
        default="post-hoc nested-CV analysis of frozen trajectories",
    )
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
