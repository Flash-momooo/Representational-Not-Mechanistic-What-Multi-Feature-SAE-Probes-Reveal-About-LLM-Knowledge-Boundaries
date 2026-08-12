"""FPE7: temporal observability, conditional information, and pre-event utility.

This experiment reuses the frozen FPE5 trajectory cache. It separates three
questions that are easy to conflate:

1. When does a realized error trajectory become distinguishable from a
   successful sibling trajectory relative to its first irreversible event?
2. Does the internal state add held-out predictive information beyond the
   prompt state, confidence statistics, and the observed token prefix?
3. How many errors can a strictly pre-event abstention policy catch at a
   calibration-selected trajectory-level false-positive rate?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from scripts.poc_fpe3_trajectory_dynamics import (
    auprc,
    auroc,
    fit_scalers,
    score_metrics,
    transform,
)
from scripts.poc_fpe5_first_divergence_hazard import load_event_table
from scripts.poc_fpe6_single_trajectory_filter import (
    ece,
    fit_pair_model,
    fit_logistic,
    logistic_predict,
    predict_pair_model,
    solve_ridge,
)


METHODS = ("question", "confidence", "observed", "dense", "sae")
UTILITY_METHODS = ("confidence", "observed", "dense", "sae")
TARGET_FPRS = (0.05, 0.10, 0.20)
ALIGN_OFFSETS = (-3, -2, -1, 0, 1, 2)
EPS = 1e-7


def group_fold_ids(pair_ids: np.ndarray, seed: int = 42, folds: int = 5) -> np.ndarray:
    unique = np.unique(pair_ids.astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {pair_id: index % folds for index, pair_id in enumerate(unique)}
    return np.asarray([mapping[value] for value in pair_ids.astype(str)], dtype=np.int8)


def feature_blocks(states: dict[str, np.ndarray], stage: int, method: str) -> list[np.ndarray]:
    raw0 = states["raw_T0_L18"].astype(np.float32)
    confidence0 = states["confidence_T0"].astype(np.float32)
    if method == "question":
        return [raw0, confidence0]

    confidence = states[f"confidence_T{stage}"].astype(np.float32)
    blocks = [raw0, confidence0]
    if stage > 0:
        blocks.append(confidence)
    if method == "confidence":
        return blocks

    blocks.append(states[f"token_prefix_T{stage}"].astype(np.float32))
    if method == "observed":
        return blocks
    if method == "dense":
        blocks.append(states[f"raw_T{stage}_L18"].astype(np.float32) - raw0)
        return blocks
    if method == "sae":
        blocks.append(
            states[f"sae_T{stage}_L18"].astype(np.float32)
            - states["sae_T0_L18"].astype(np.float32)
        )
        return blocks
    raise ValueError(method)


def fit_model(
    blocks: list[np.ndarray], labels: np.ndarray, train: np.ndarray, device: str
) -> dict:
    if len(np.unique(labels[train])) < 2:
        probability = float(np.mean(labels[train]))
        return {"constant": probability}
    scalers = fit_scalers(blocks, train)
    features = transform(blocks, scalers)
    intercept, weights = solve_ridge(features[train], labels[train], device)
    raw_train = intercept + features[train] @ weights
    calibrator = fit_logistic(raw_train[:, None], labels[train], ridge=0.02)
    return {
        "constant": None,
        "scalers": scalers,
        "intercept": intercept,
        "weights": weights,
        "calibrator": calibrator,
    }


def predict_model(model: dict, blocks: list[np.ndarray]) -> np.ndarray:
    if model["constant"] is not None:
        return np.full(len(blocks[0]), model["constant"], dtype=np.float64)
    features = transform(blocks, model["scalers"])
    raw = model["intercept"] + features @ model["weights"]
    return logistic_predict(model["calibrator"], raw[:, None])


def state_delta_blocks(
    states: dict[str, np.ndarray], stage: int, method: str
) -> list[np.ndarray]:
    if method == "dense":
        return [
            states[f"raw_T{stage}_L18"].astype(np.float32)
            - states["raw_T0_L18"].astype(np.float32)
        ]
    if method == "sae":
        return [
            states[f"sae_T{stage}_L18"].astype(np.float32)
            - states["sae_T0_L18"].astype(np.float32)
        ]
    raise ValueError(method)


def paired_state_blocks(
    states: dict[str, np.ndarray], stage: int, method: str
) -> list[np.ndarray]:
    return state_delta_blocks(states, stage, method) + [
        states[f"token_prefix_T{stage}"].astype(np.float32),
        states[f"confidence_T{stage}"].astype(np.float32),
    ]


def fit_method_model(
    states: dict[str, np.ndarray],
    stage: int,
    method: str,
    labels: np.ndarray,
    train: np.ndarray,
    device: str,
    pair_state: bool = True,
) -> dict:
    if method not in ("dense", "sae") or stage == 0 or not pair_state:
        return {
            "kind": "ridge",
            "model": fit_model(feature_blocks(states, stage, method), labels, train, device),
        }

    # The paired readout learns only from within-question differences. Dense
    # and sparse states receive exactly the same token/confidence covariates.
    state_blocks = paired_state_blocks(states, stage, method)
    state_model = fit_pair_model(
        state_blocks,
        labels,
        states["question_ids"].astype(str),
        train,
        device,
    )
    return {"kind": "pair", "state": state_model}


def predict_method_model(
    model: dict, states: dict[str, np.ndarray], stage: int, method: str
) -> np.ndarray:
    if model["kind"] == "ridge":
        return predict_model(model["model"], feature_blocks(states, stage, method))
    return predict_pair_model(
        model["state"], paired_state_blocks(states, stage, method)
    )


def cross_validated_predictions(
    states: dict[str, np.ndarray],
    labels: np.ndarray,
    valid: np.ndarray,
    stage: int,
    fold_ids: np.ndarray,
    methods: tuple[str, ...],
    device: str,
) -> dict[str, np.ndarray]:
    output = {
        method: np.full(len(labels), np.nan, dtype=np.float64) for method in methods
    }
    for fold in range(5):
        train = valid & (fold_ids != fold)
        test = valid & (fold_ids == fold)
        if not test.any():
            continue
        for method in methods:
            model = fit_method_model(states, stage, method, labels, train, device)
            output[method][test] = predict_method_model(
                model, states, stage, method
            )[test]
    return output


def event_onsets(table: dict[str, np.ndarray]) -> np.ndarray:
    onset = np.full(table["event"].shape[1], -1, dtype=np.int8)
    for stage in range(table["event"].shape[0]):
        selected = (onset < 0) & table["risk"][stage] & (table["event"][stage] == 1)
        onset[selected] = stage
    return onset


def bootstrap(values: np.ndarray, seed: int, reference: float = 0.0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": [None, None], "n_groups": 0}
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(sampled, (0.025, 0.975))],
        "probability_at_or_below_reference": float(np.mean(sampled <= reference)),
        "n_groups": int(len(values)),
    }


def final_outcome_oof(
    states: dict[str, np.ndarray],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    max_stage: int,
    device: str,
) -> dict[tuple[int, str], np.ndarray]:
    predictions = {}
    for stage in range(max_stage + 1):
        valid = states[f"valid_T{stage}"].astype(bool)
        stage_predictions = cross_validated_predictions(
            states, labels, valid, stage, fold_ids, METHODS, device
        )
        for method, values in stage_predictions.items():
            predictions[(stage, method)] = values
    return predictions


def temporal_alignment(
    states: dict[str, np.ndarray],
    labels: np.ndarray,
    onsets: np.ndarray,
    predictions: dict[tuple[int, str], np.ndarray],
    max_stage: int,
) -> list[dict]:
    question_ids = states["question_ids"].astype(str)
    rows = []
    for offset in ALIGN_OFFSETS:
        for method in METHODS:
            by_question = defaultdict(list)
            n_pairs = 0
            n_events = 0
            for error_row in np.flatnonzero(onsets >= 0):
                stage = int(onsets[error_row]) + offset
                if stage < 0 or stage > max_stage:
                    continue
                if not states[f"valid_T{stage}"][error_row]:
                    continue
                scores = predictions[(stage, method)]
                controls = (
                    (question_ids == question_ids[error_row])
                    & (labels == 0)
                    & states[f"valid_T{stage}"].astype(bool)
                    & np.isfinite(scores)
                )
                if not controls.any() or not np.isfinite(scores[error_row]):
                    continue
                differences = scores[error_row] - scores[controls]
                pair_auc = float(np.mean((differences > 0) + 0.5 * (differences == 0)))
                by_question[question_ids[error_row]].append(pair_auc)
                n_pairs += int(controls.sum())
                n_events += 1
            question_values = np.asarray(
                [np.mean(values) for values in by_question.values()], dtype=np.float64
            )
            estimate = bootstrap(question_values, 2026071600 + 20 * (offset + 3) + METHODS.index(method), 0.5)
            rows.append({
                "offset_from_pre_event_state": offset,
                "interpretation": (
                    "state immediately before the error transition" if offset == 0
                    else "state after the error token" if offset == 1
                    else "event-relative state"
                ),
                "method": method,
                "paired_sibling_auroc_macro": estimate["mean"],
                "ci95": estimate["ci95"],
                "n_questions": estimate["n_groups"],
                "n_event_trajectories": n_events,
                "n_error_success_pairs": n_pairs,
            })
    return rows


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = np.clip(probabilities, EPS, 1.0 - EPS)
    return {
        "auroc": auroc(labels, probabilities),
        "auprc": auprc(labels, probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll_nats": float(np.mean(
            -labels * np.log(probabilities) - (1 - labels) * np.log(1 - probabilities)
        )),
        "ece10": ece(labels, probabilities),
    }


def grouped_loss_delta(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> dict:
    left = np.clip(left, EPS, 1.0 - EPS)
    right = np.clip(right, EPS, 1.0 - EPS)
    left_loss = -labels * np.log(left) - (1 - labels) * np.log(1 - left)
    right_loss = -labels * np.log(right) - (1 - labels) * np.log(1 - right)
    values = []
    for group in np.unique(groups[valid]):
        selected = valid & (groups == group)
        values.append(float(np.mean(right_loss[selected] - left_loss[selected]) / math.log(2.0)))
    return bootstrap(np.asarray(values), seed, 0.0)


def grouped_pairwise_loss_delta(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    question_ids: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> dict:
    left_logits = np.log(np.clip(left, EPS, 1 - EPS) / np.clip(1 - left, EPS, 1 - EPS))
    right_logits = np.log(np.clip(right, EPS, 1 - EPS) / np.clip(1 - right, EPS, 1 - EPS))
    values = []
    n_pairs = 0
    for question in np.unique(question_ids[valid]):
        local = valid & (question_ids == question)
        positive = np.flatnonzero(local & (labels == 1))
        negative = np.flatnonzero(local & (labels == 0))
        if not len(positive) or not len(negative):
            continue
        left_difference = (
            left_logits[positive, None] - left_logits[negative][None, :]
        ).reshape(-1)
        right_difference = (
            right_logits[positive, None] - right_logits[negative][None, :]
        ).reshape(-1)
        left_loss = np.logaddexp(0.0, -left_difference)
        right_loss = np.logaddexp(0.0, -right_difference)
        values.append(float(np.mean(right_loss - left_loss) / math.log(2.0)))
        n_pairs += len(left_loss)
    estimate = bootstrap(np.asarray(values), seed, 0.0)
    estimate["n_positive_negative_pairs"] = int(n_pairs)
    return estimate


def information_increment(
    states: dict[str, np.ndarray],
    table: dict[str, np.ndarray],
    fold_ids: np.ndarray,
    max_stage: int,
    device: str,
) -> tuple[list[dict], list[dict], dict[tuple[int, str], np.ndarray]]:
    pair_ids = states["pair_ids"].astype(str)
    question_ids = states["question_ids"].astype(str)
    summaries = []
    contrasts = []
    all_predictions = {}
    comparisons = (
        ("confidence", "question", "confidence beyond prompt state"),
        ("observed", "confidence", "token prefix beyond confidence"),
        ("dense", "observed", "dense state beyond observed information"),
        ("sae", "observed", "SAE state beyond observed information"),
        ("sae", "dense", "SAE versus dense state"),
    )
    for stage in range(max_stage + 1):
        valid = table["risk"][stage]
        labels = table["event"][stage].astype(np.int8)
        predictions = cross_validated_predictions(
            states, labels, valid, stage, fold_ids, METHODS, device
        )
        for method, values in predictions.items():
            selected = valid & np.isfinite(values)
            metric = binary_metrics(labels[selected], values[selected])
            within, _ = score_metrics(labels, values, question_ids, selected)
            summaries.append({
                "stage": f"T{stage}",
                "transition": f"T{stage}->T{stage + 1}",
                "method": method,
                **metric,
                "within_question_auroc_macro": within["within_question_auroc_macro"],
                "n_rows": int(selected.sum()),
                "n_events": int(labels[selected].sum()),
            })
            all_predictions[(stage, method)] = values
        for left, right, description in comparisons:
            population_estimate = grouped_loss_delta(
                labels,
                predictions[left],
                predictions[right],
                pair_ids,
                valid,
                2026071700 + stage * 20 + comparisons.index((left, right, description)),
            )
            estimate = grouped_pairwise_loss_delta(
                labels,
                predictions[left],
                predictions[right],
                question_ids,
                valid,
                2026071800 + stage * 20 + comparisons.index((left, right, description)),
            )
            contrasts.append({
                "stage": f"T{stage}",
                "contrast": f"{left}_minus_{right}",
                "description": description,
                "heldout_information_gain_bits_per_transition": estimate["mean"],
                "ci95": estimate["ci95"],
                "probability_gain_at_or_below_zero": estimate.get(
                    "probability_at_or_below_reference"
                ),
                "n_pair_groups": estimate["n_groups"],
                "n_positive_negative_pairs": estimate["n_positive_negative_pairs"],
                "population_logloss_gain_bits_per_transition": population_estimate["mean"],
                "population_logloss_gain_ci95": population_estimate["ci95"],
            })
    return summaries, contrasts, all_predictions


def quantile_threshold(values: np.ndarray, target_fpr: float) -> float:
    if len(values) == 0:
        return 1.0
    return float(np.quantile(values, 1.0 - target_fpr, method="higher"))


def trajectory_max_scores(
    scores: dict[tuple[int, str], np.ndarray],
    table: dict[str, np.ndarray],
    rows: np.ndarray,
    method: str,
    max_stage: int,
) -> np.ndarray:
    output = np.full(table["risk"].shape[1], -np.inf, dtype=np.float64)
    for stage in range(max_stage + 1):
        valid = rows & table["risk"][stage] & np.isfinite(scores[(stage, method)])
        output[valid] = np.maximum(output[valid], scores[(stage, method)][valid])
    return output


def nested_pre_event_utility(
    states: dict[str, np.ndarray],
    labels: np.ndarray,
    table: dict[str, np.ndarray],
    onsets: np.ndarray,
    fold_ids: np.ndarray,
    max_stage: int,
    device: str,
) -> list[dict]:
    records = []
    n = len(labels)
    for fold in range(5):
        test_rows = fold_ids == fold
        calibration_rows = fold_ids == ((fold + 1) % 5)
        train_rows = ~(test_rows | calibration_rows)
        fold_scores = {}
        for stage in range(max_stage + 1):
            valid = table["risk"][stage]
            hazard_labels = table["event"][stage].astype(np.int8)
            for method in UTILITY_METHODS:
                model = fit_method_model(
                    states, stage, method, hazard_labels, valid & train_rows, device,
                    pair_state=False,
                )
                predictions = predict_method_model(model, states, stage, method)
                fold_scores[(stage, method)] = predictions

        for method in UTILITY_METHODS:
            calibration_correct = calibration_rows & (labels == 0)
            calibration_max = trajectory_max_scores(
                fold_scores, table, calibration_correct, method, max_stage
            )
            calibration_max = calibration_max[np.isfinite(calibration_max)]
            for target_fpr in TARGET_FPRS:
                threshold = quantile_threshold(calibration_max, target_fpr)
                alert_stage = np.full(n, -1, dtype=np.int8)
                for stage in range(max_stage + 1):
                    eligible = (
                        test_rows
                        & table["risk"][stage]
                        & (alert_stage < 0)
                        & (fold_scores[(stage, method)] >= threshold)
                    )
                    alert_stage[eligible] = stage

                test_correct = test_rows & (labels == 0)
                test_wrong_observed = test_rows & (labels == 1) & (onsets >= 0)
                false_alert = test_correct & (alert_stage >= 0)
                detected = (
                    test_wrong_observed
                    & (alert_stage >= 0)
                    & (alert_stage <= onsets)
                )
                lead = onsets[detected] - alert_stage[detected] + 1
                records.append({
                    "fold": fold,
                    "method": method,
                    "target_trajectory_fpr": target_fpr,
                    "threshold_from_calibration_fold": threshold,
                    "n_test": int(test_rows.sum()),
                    "n_test_correct": int(test_correct.sum()),
                    "n_test_wrong_with_observed_event": int(test_wrong_observed.sum()),
                    "false_alerts": int(false_alert.sum()),
                    "detected_errors_before_token": int(detected.sum()),
                    "actual_trajectory_fpr": float(false_alert.sum() / max(test_correct.sum(), 1)),
                    "pre_event_recall": float(detected.sum() / max(test_wrong_observed.sum(), 1)),
                    "mean_token_lead": float(np.mean(lead)) if len(lead) else None,
                    "utility_cost_0_25": float((detected.sum() - 0.25 * false_alert.sum()) / max(test_rows.sum(), 1)),
                    "utility_cost_1_00": float((detected.sum() - false_alert.sum()) / max(test_rows.sum(), 1)),
                })

    summary = []
    for method in UTILITY_METHODS:
        for target_fpr in TARGET_FPRS:
            selected = [
                row for row in records
                if row["method"] == method
                and row["target_trajectory_fpr"] == target_fpr
            ]
            totals = {
                key: sum(row[key] for row in selected)
                for key in (
                    "n_test", "n_test_correct", "n_test_wrong_with_observed_event",
                    "false_alerts", "detected_errors_before_token",
                )
            }
            lead_numerator = sum(
                row["mean_token_lead"] * row["detected_errors_before_token"]
                for row in selected if row["mean_token_lead"] is not None
            )
            summary.append({
                "method": method,
                "target_trajectory_fpr": target_fpr,
                **totals,
                "actual_trajectory_fpr": float(
                    totals["false_alerts"] / max(totals["n_test_correct"], 1)
                ),
                "pre_event_recall": float(
                    totals["detected_errors_before_token"]
                    / max(totals["n_test_wrong_with_observed_event"], 1)
                ),
                "mean_token_lead": float(
                    lead_numerator / max(totals["detected_errors_before_token"], 1)
                ),
                "utility_cost_0_25": float(
                    (totals["detected_errors_before_token"] - 0.25 * totals["false_alerts"])
                    / max(totals["n_test"], 1)
                ),
                "utility_cost_1_00": float(
                    (totals["detected_errors_before_token"] - totals["false_alerts"])
                    / max(totals["n_test"], 1)
                ),
            })
    return summary


def make_figures(payload: dict, output_dir: Path, prefix: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    colors = {
        "question": "#777777",
        "confidence": "#D55E00",
        "observed": "#CC79A7",
        "dense": "#0072B2",
        "sae": "#009E73",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in METHODS:
        selected = [row for row in payload["temporal_alignment"] if row["method"] == method]
        x = [row["offset_from_pre_event_state"] for row in selected]
        y = [row["paired_sibling_auroc_macro"] for row in selected]
        ax.plot(x, y, marker="o", linewidth=2, label=method, color=colors[method])
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.axvline(0, color="#333333", linestyle=":", linewidth=1)
    ax.set_xlabel("State offset from first-error transition (0 = before token)")
    ax.set_ylabel("Paired sibling AUROC")
    ax.set_ylim(0.35, 1.02)
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{prefix}_observability_alignment.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    contrasts = ("confidence_minus_question", "observed_minus_confidence", "dense_minus_observed", "sae_minus_observed")
    labels = ("Confidence | X", "Prefix | confidence", "Dense | observed", "SAE | observed")
    x = np.arange(payload["max_stage"] + 1)
    width = 0.19
    for index, (contrast, label) in enumerate(zip(contrasts, labels)):
        selected = [row for row in payload["information_contrasts"] if row["contrast"] == contrast]
        y = [row["heldout_information_gain_bits_per_transition"] for row in selected]
        ax.bar(x + (index - 1.5) * width, y, width, label=label)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, [f"T{stage}" for stage in x])
    ax.set_ylabel("Conditional pairwise information gain (bits / pair)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{prefix}_information_increment.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method in UTILITY_METHODS:
        selected = [row for row in payload["pre_event_utility"] if row["method"] == method]
        ax.plot(
            [row["actual_trajectory_fpr"] for row in selected],
            [row["pre_event_recall"] for row in selected],
            marker="o", linewidth=2, label=method, color=colors[method],
        )
    ax.set_xlabel("Held-out trajectory false-positive rate")
    ax.set_ylabel("Pre-event recall")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{prefix}_pre_event_utility.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(str(path))
    return paths


def render_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# FPE7 Observability and Pre-Event Utility Results",
        "",
        f"Dataset: `{payload['data_path']}`  ",
        f"Model: `{payload['model']}`  ",
        f"Maximum evaluated pre-transition stage: T{payload['max_stage']}",
        "",
        "## Temporal Alignment",
        "",
        "Paired sibling AUROC compares an error trajectory with successful completions of the same question.",
        "Offset 0 is the state immediately before the first irreversible error token; offset +1 is after it.",
        "",
        "| Offset | Question | Confidence | Observed | Dense | SAE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for offset in ALIGN_OFFSETS:
        values = {
            row["method"]: row["paired_sibling_auroc_macro"]
            for row in payload["temporal_alignment"]
            if row["offset_from_pre_event_state"] == offset
        }
        lines.append(
            f"| {offset:+d} | " + " | ".join(
                "NA" if values.get(method) is None else f"{values[method]:.3f}"
                for method in METHODS
            ) + " |"
        )
    lines.extend([
        "", "## Conditional Information", "",
        "Positive values mean that the left model reduces held-out within-question pairwise log loss beyond the right model.",
        "", "| Stage | Confidence beyond X | Prefix beyond confidence | Dense beyond observed | SAE beyond observed |",
        "|---|---:|---:|---:|---:|",
    ])
    for stage in range(payload["max_stage"] + 1):
        values = {
            row["contrast"]: row["heldout_information_gain_bits_per_transition"]
            for row in payload["information_contrasts"] if row["stage"] == f"T{stage}"
        }
        lines.append(
            f"| T{stage} | {values['confidence_minus_question']:.4f} | "
            f"{values['observed_minus_confidence']:.4f} | "
            f"{values['dense_minus_observed']:.4f} | "
            f"{values['sae_minus_observed']:.4f} |"
        )
    lines.extend([
        "", "## Strictly Pre-Event Abstention", "",
        "Thresholds are selected on a separate calibration fold using successful trajectories.",
        "", "| Method | Target FPR | Actual FPR | Pre-event recall | Mean token lead | Utility (FP cost 0.25) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["pre_event_utility"]:
        lines.append(
            f"| {row['method']} | {row['target_trajectory_fpr']:.0%} | "
            f"{row['actual_trajectory_fpr']:.3f} | {row['pre_event_recall']:.3f} | "
            f"{row['mean_token_lead']:.2f} | {row['utility_cost_0_25']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], local_files_only=True
    )
    data_path = ROOT / args.data
    cache_path = ROOT / args.cache
    rows, states, event_data = load_event_table(tokenizer, data_path, cache_path)
    del rows
    table = event_data["table"]
    labels = states["labels"].astype(np.int8)
    onsets = event_onsets(table)
    fold_ids = group_fold_ids(states["pair_ids"].astype(str), seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    final_predictions = final_outcome_oof(
        states, labels, fold_ids, args.max_stage, device
    )
    alignment = temporal_alignment(
        states, labels, onsets, final_predictions, args.max_stage
    )
    information, contrasts, _ = information_increment(
        states, table, fold_ids, args.max_stage, device
    )
    utility = nested_pre_event_utility(
        states, labels, table, onsets, fold_ids, args.max_stage, device
    )

    payload = {
        "experiment": "FPE7 temporal observability and strictly pre-event utility",
        "model": config["model"]["name"],
        "data_path": str(data_path),
        "cache_path": str(cache_path),
        "max_stage": args.max_stage,
        "seed": args.seed,
        "device": device,
        "protocol": {
            "event": "first irreversible normalized exact-answer divergence",
            "outer_split": "five pair_id-grouped folds",
            "utility_threshold": "separate rotating calibration fold; test untouched",
            "intervention": "abstain only from states at or before the error transition",
            "information_unit": "held-out within-question pairwise cross-entropy reduction in bits",
        },
        "label_agreement": event_data["agreement"],
        "n_rows": len(labels),
        "n_observed_error_events": int(np.sum(onsets >= 0)),
        "temporal_alignment": alignment,
        "information_summary": information,
        "information_contrasts": contrasts,
        "pre_event_utility": utility,
    }
    output_path = ROOT / args.output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["figures"] = make_figures(payload, ROOT / "paper" / "figures", args.prefix)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path = ROOT / "paper" / f"{args.prefix.upper()}_OBSERVABILITY_UTILITY_RESULTS.md"
    render_markdown(payload, markdown_path)
    print(f"saved -> {output_path}")
    print(f"saved -> {markdown_path}")
    for row in contrasts:
        if row["stage"] in ("T0", "T1"):
            print(
                f"{row['stage']} {row['contrast']:<28} "
                f"gain={row['heldout_information_gain_bits_per_transition']:.5f} bits"
            )
    for row in utility:
        if row["target_trajectory_fpr"] == 0.10:
            print(
                f"utility {row['method']:<10} fpr={row['actual_trajectory_fpr']:.3f} "
                f"recall={row['pre_event_recall']:.3f} lead={row['mean_token_lead']:.2f}"
            )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v37_2wiki_trajectory.jsonl")
    parser.add_argument("--cache", default="outputs/cache/v37_2wiki_trajectory_states.npz")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/poc_fpe7_observability_utility_results.json")
    parser.add_argument("--prefix", default="fpe7")
    parser.add_argument("--max-stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
