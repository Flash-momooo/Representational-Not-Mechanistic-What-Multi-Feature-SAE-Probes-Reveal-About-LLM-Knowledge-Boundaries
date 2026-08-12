"""FPE9: grouped cross-domain calibration of trajectory risk scales.

FPE8c showed that several monitors retain useful HotpotQA ranking while their
source-domain operating thresholds do not transfer. This experiment keeps the
monitor fixed and measures how many labeled target trajectories are required
to recover a trajectory-level false-positive-rate operating point.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "cache" / "fpe8c_cross_domain_scores.npz"
OUTPUT = ROOT / "outputs" / "poc_fpe9_cross_domain_risk_calibration_results.json"
REPORT = ROOT / "paper" / "FPE9_CROSS_DOMAIN_RISK_CALIBRATION_RESULTS.md"
FIGURE = ROOT / "paper" / "figures" / "fpe9_calibration_curve.png"

METHODS = ("teacher", "observed", "direct_dynamic", "fixed_dynamic", "fixed_shared")
METHOD_LABELS = {
    "teacher": "Dense teacher",
    "observed": "Token-confidence baseline",
    "direct_dynamic": "Sparse dynamic K=8",
    "fixed_dynamic": "Distilled sparse K=8",
    "fixed_shared": "Shared sparse K=8",
}
STAGES = (1, 2, 3)
TARGET_FPR = 0.10
PAIR_SIZES = (1, 2, 4, 8, 12, 16, 24, 40)
N_SEEDS = 100
EPS = 1e-7


def stage_scores(cache, domain: str, method: str) -> dict[int, np.ndarray]:
    return {
        stage: cache[f"{domain}_score_{method}_T{stage}"].astype(np.float64)
        for stage in STAGES
    }


def trajectory_max(scores: dict[int, np.ndarray], risk: np.ndarray) -> np.ndarray:
    output = np.full(risk.shape[1], -np.inf, dtype=np.float64)
    for stage in STAGES:
        valid = risk[stage].astype(bool)
        output[valid] = np.maximum(output[valid], scores[stage][valid])
    return output


def higher_quantile(values: np.ndarray, level: float) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    return float(np.quantile(values, np.clip(level, 0.0, 1.0), method="higher"))


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            value += selected.mean() * abs(labels[selected].mean() - probability[selected].mean())
    return float(value) if total else math.nan


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability.astype(np.float64), EPS, 1 - EPS)
    result = {
        "brier": float(np.mean((probability - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))),
        "ece_10": ece(labels, probability),
        "auroc": math.nan,
        "auprc": math.nan,
    }
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, probability))
        result["auprc"] = float(average_precision_score(labels, probability))
    return result


def fit_platt(scores: np.ndarray, labels: np.ndarray):
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if len(np.unique(labels)) < 2:
        return None
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
    model.fit(scores[:, None], labels)
    slope = max(float(model.coef_[0, 0]), EPS)
    intercept = float(model.intercept_[0])
    if model.coef_[0, 0] <= 0:
        prevalence = np.clip(labels.mean(), EPS, 1 - EPS)
        intercept = float(np.log(prevalence / (1 - prevalence)) - slope * scores.mean())
    return {"slope": slope, "intercept": intercept}


def refit_intercept(model: dict | None, scores: np.ndarray, labels: np.ndarray):
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    if model is None or len(np.unique(labels)) < 2:
        return None
    target = float(labels.mean())
    slope = model["slope"]
    lo, hi = -30.0, 30.0
    for _ in range(80):
        midpoint = (lo + hi) / 2
        probability = 1 / (1 + np.exp(-np.clip(midpoint + slope * scores, -30, 30)))
        if probability.mean() < target:
            lo = midpoint
        else:
            hi = midpoint
    return {"slope": slope, "intercept": (lo + hi) / 2}


def platt_predict(model, scores: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(scores), np.nan)
    linear = model["intercept"] + model["slope"] * scores
    return 1 / (1 + np.exp(-np.clip(linear, -30, 30)))


def threshold_metrics(
    threshold: float,
    scores: dict[int, np.ndarray],
    risk: np.ndarray,
    labels: np.ndarray,
    onsets: np.ndarray,
    evaluation: np.ndarray,
) -> dict[str, float | int | None]:
    if not np.isfinite(threshold) and threshold != math.inf:
        return {
            "n_correct": int((evaluation & (labels == 0)).sum()),
            "n_eligible_error": int((evaluation & (labels == 1) & (onsets >= 1)).sum()),
            "actual_fpr": math.nan,
            "abs_fpr_error": math.nan,
            "pre_event_recall": math.nan,
            "mean_stage_lead": None,
        }
    alert_stage = np.full(len(labels), -1, dtype=np.int8)
    for stage in STAGES:
        selected = evaluation & risk[stage].astype(bool) & (alert_stage < 0)
        selected &= scores[stage] >= threshold
        alert_stage[selected] = stage
    correct = evaluation & (labels == 0)
    eligible = evaluation & (labels == 1) & (onsets >= 1)
    false_alert = correct & (alert_stage >= 0)
    detected = eligible & (alert_stage >= 0) & (alert_stage <= onsets)
    leads = onsets[detected] - alert_stage[detected] + 1
    actual_fpr = false_alert.sum() / max(correct.sum(), 1)
    recall = detected.sum() / max(eligible.sum(), 1)
    return {
        "n_correct": int(correct.sum()),
        "n_eligible_error": int(eligible.sum()),
        "actual_fpr": float(actual_fpr),
        "abs_fpr_error": float(abs(actual_fpr - TARGET_FPR)),
        "pre_event_recall": float(recall),
        "mean_stage_lead": float(leads.mean()) if len(leads) else None,
    }


def affine_threshold(source_correct: np.ndarray, target_correct: np.ndarray, source_threshold: float) -> float:
    if not len(target_correct):
        return math.nan
    source_median = float(np.median(source_correct))
    target_median = float(np.median(target_correct))
    if len(target_correct) < 4:
        return source_threshold + target_median - source_median
    source_iqr = float(np.quantile(source_correct, 0.75) - np.quantile(source_correct, 0.25))
    target_iqr = float(np.quantile(target_correct, 0.75) - np.quantile(target_correct, 0.25))
    if source_iqr <= EPS or target_iqr <= EPS:
        return source_threshold + target_median - source_median
    scale = float(np.clip(target_iqr / source_iqr, 0.25, 4.0))
    return target_median + scale * (source_threshold - source_median)


def aggregate(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    result = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[metric] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "p05": float(np.quantile(finite, 0.05)) if len(finite) else None,
            "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
            "n_valid": int(len(finite)),
        }
    return result


def run() -> dict:
    cache = np.load(CACHE)
    source_labels = cache["source_labels"].astype(np.int8)
    target_labels = cache["target_labels"].astype(np.int8)
    target_onsets = cache["target_onsets"].astype(np.int8)
    source_risk = cache["source_risk"].astype(bool)
    target_risk = cache["target_risk"].astype(bool)
    source_calibration = cache["source_calibration_rows"].astype(bool)
    target_pairs = cache["target_pair_ids"].astype(str)
    unique_pairs = np.unique(target_pairs)

    threshold_runs = []
    calibration_runs = []
    method_metadata = {}
    for method in METHODS:
        source_scores = stage_scores(cache, "source", method)
        target_scores = stage_scores(cache, "target", method)
        source_max = trajectory_max(source_scores, source_risk)
        target_max = trajectory_max(target_scores, target_risk)
        source_correct = source_max[source_calibration & (source_labels == 0)]
        source_threshold = higher_quantile(source_correct, 1 - TARGET_FPR)
        source_all = source_calibration & np.isfinite(source_max)
        source_alert_rate = float(np.mean(source_max[source_all] >= source_threshold))
        source_platt = fit_platt(source_max[source_all], source_labels[source_all])
        oracle_threshold = higher_quantile(target_max[target_labels == 0], 1 - TARGET_FPR)
        method_metadata[method] = {
            "source_threshold": source_threshold,
            "source_alert_rate": source_alert_rate,
            "n_source_calibration": int(source_all.sum()),
            "n_source_calibration_correct": int(len(source_correct)),
            "oracle_target_threshold": oracle_threshold,
        }

        for seed in range(N_SEEDS):
            rng = np.random.default_rng(91000 + seed)
            shuffled = unique_pairs.copy()
            rng.shuffle(shuffled)
            for n_pairs in PAIR_SIZES:
                calibration_pairs = shuffled[:n_pairs]
                calibration = np.isin(target_pairs, calibration_pairs)
                evaluation = ~calibration
                calibration_correct = calibration & (target_labels == 0) & np.isfinite(target_max)
                target_correct_scores = target_max[calibration_correct]
                target_all_scores = target_max[calibration & np.isfinite(target_max)]

                thresholds = {
                    "source_fixed": source_threshold,
                    "unlabeled_rate_match": higher_quantile(target_all_scores, 1 - source_alert_rate),
                    "target_empirical": higher_quantile(target_correct_scores, 1 - TARGET_FPR),
                    "target_affine": affine_threshold(source_correct, target_correct_scores, source_threshold),
                    "target_conformal": math.inf,
                    "oracle_full_target": oracle_threshold,
                }
                n_correct = len(target_correct_scores)
                conformal_rank = int(math.ceil((n_correct + 1) * (1 - TARGET_FPR)))
                if n_correct and conformal_rank <= n_correct:
                    thresholds["target_conformal"] = float(np.sort(target_correct_scores)[conformal_rank - 1])
                for calibration_method, threshold in thresholds.items():
                    threshold_runs.append({
                        "method": method,
                        "calibration_method": calibration_method,
                        "seed": seed,
                        "n_pairs": n_pairs,
                        "n_calibration": int(calibration.sum()),
                        "n_calibration_correct": int(calibration_correct.sum()),
                        "threshold": threshold if np.isfinite(threshold) else None,
                        **threshold_metrics(
                            threshold, target_scores, target_risk, target_labels,
                            target_onsets, evaluation,
                        ),
                    })

                eval_finite = evaluation & np.isfinite(target_max)
                if not eval_finite.any():
                    continue
                probability_models = {
                    "raw_score": np.clip(target_max[eval_finite], EPS, 1 - EPS),
                    "source_platt": platt_predict(source_platt, target_max[eval_finite]),
                }
                target_platt = fit_platt(target_max[calibration], target_labels[calibration])
                intercept_only = refit_intercept(
                    source_platt, target_max[calibration], target_labels[calibration]
                )
                if target_platt is not None:
                    probability_models["target_platt"] = platt_predict(
                        target_platt, target_max[eval_finite]
                    )
                if intercept_only is not None:
                    probability_models["target_intercept"] = platt_predict(
                        intercept_only, target_max[eval_finite]
                    )
                for calibration_method, probability in probability_models.items():
                    if not np.isfinite(probability).all():
                        continue
                    calibration_runs.append({
                        "method": method,
                        "calibration_method": calibration_method,
                        "seed": seed,
                        "n_pairs": n_pairs,
                        "n_calibration": int(calibration.sum()),
                        "n_calibration_correct": int(calibration_correct.sum()),
                        **probability_metrics(target_labels[eval_finite], probability),
                    })

    threshold_summary = []
    for key in sorted({(row["method"], row["calibration_method"], row["n_pairs"]) for row in threshold_runs}):
        method, calibration_method, n_pairs = key
        rows = [row for row in threshold_runs if (row["method"], row["calibration_method"], row["n_pairs"]) == key]
        threshold_summary.append({
            "method": method,
            "calibration_method": calibration_method,
            "n_pairs": n_pairs,
            "n_calibration": n_pairs * 8,
            "n_seeds": len(rows),
            "mean_calibration_correct": float(np.mean([row["n_calibration_correct"] for row in rows])),
            **aggregate(rows, ("actual_fpr", "abs_fpr_error", "pre_event_recall")),
        })

    probability_summary = []
    for key in sorted({(row["method"], row["calibration_method"], row["n_pairs"]) for row in calibration_runs}):
        method, calibration_method, n_pairs = key
        rows = [row for row in calibration_runs if (row["method"], row["calibration_method"], row["n_pairs"]) == key]
        probability_summary.append({
            "method": method,
            "calibration_method": calibration_method,
            "n_pairs": n_pairs,
            "n_calibration": n_pairs * 8,
            "n_seeds": len(rows),
            **aggregate(rows, ("auroc", "auprc", "brier", "nll", "ece_10")),
        })

    paired_comparisons = []
    for method in ("observed", "direct_dynamic", "fixed_dynamic", "fixed_shared"):
        for calibration_method in (
            "source_fixed", "unlabeled_rate_match", "target_empirical",
            "target_affine", "target_conformal", "oracle_full_target",
        ):
            for n_pairs in PAIR_SIZES:
                candidate = {
                    row["seed"]: row for row in threshold_runs
                    if row["method"] == method
                    and row["calibration_method"] == calibration_method
                    and row["n_pairs"] == n_pairs
                }
                teacher = {
                    row["seed"]: row for row in threshold_runs
                    if row["method"] == "teacher"
                    and row["calibration_method"] == calibration_method
                    and row["n_pairs"] == n_pairs
                }
                shared_seeds = sorted(set(candidate) & set(teacher))
                recall_delta = np.asarray([
                    candidate[seed]["pre_event_recall"] - teacher[seed]["pre_event_recall"]
                    for seed in shared_seeds
                ], dtype=np.float64)
                fpr_delta = np.asarray([
                    candidate[seed]["actual_fpr"] - teacher[seed]["actual_fpr"]
                    for seed in shared_seeds
                ], dtype=np.float64)
                finite = np.isfinite(recall_delta) & np.isfinite(fpr_delta)
                recall_delta = recall_delta[finite]
                fpr_delta = fpr_delta[finite]
                paired_comparisons.append({
                    "method": method,
                    "reference": "teacher",
                    "calibration_method": calibration_method,
                    "n_pairs": n_pairs,
                    "n_calibration": n_pairs * 8,
                    "n_valid": int(len(recall_delta)),
                    "recall_delta_mean": float(recall_delta.mean()) if len(recall_delta) else None,
                    "recall_delta_p05": float(np.quantile(recall_delta, 0.05)) if len(recall_delta) else None,
                    "recall_delta_p95": float(np.quantile(recall_delta, 0.95)) if len(recall_delta) else None,
                    "fpr_delta_mean": float(fpr_delta.mean()) if len(fpr_delta) else None,
                    "dominance_rate": float(np.mean((recall_delta >= 0) & (fpr_delta <= 0))) if len(recall_delta) else None,
                })

    payload = {
        "experiment": "FPE9 grouped cross-domain trajectory risk calibration",
        "source": "pooled 2Wiki A+B",
        "target": "untouched HotpotQA C1",
        "target_fpr": TARGET_FPR,
        "n_seeds": N_SEEDS,
        "calibration_pair_sizes": list(PAIR_SIZES),
        "calibration_trajectory_sizes": [8 * value for value in PAIR_SIZES],
        "grouping": "pair_id; all eight trajectories for a question remain together",
        "method_labels": METHOD_LABELS,
        "method_metadata": method_metadata,
        "threshold_summary": threshold_summary,
        "probability_summary": probability_summary,
        "paired_comparisons": paired_comparisons,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_report(payload)
    plot_results(payload)
    print(f"saved -> {OUTPUT}")
    print(f"report -> {REPORT}")
    print(f"figure -> {FIGURE}")
    print_key_results(payload)
    return payload


def summary_row(payload: dict, method: str, calibration_method: str, n_calibration: int) -> dict:
    return next(
        row for row in payload["threshold_summary"]
        if row["method"] == method
        and row["calibration_method"] == calibration_method
        and row["n_calibration"] == n_calibration
    )


def write_report(payload: dict) -> None:
    lines = [
        "# FPE9 Cross-Domain Risk Calibration",
        "",
        "## Protocol",
        "",
        "- Frozen monitors: pooled 2Wiki A+B models from FPE8c; no monitor weights are updated.",
        "- Target: HotpotQA, split by question-level `pair_id`; eight sibling trajectories never cross calibration/test.",
        "- Repeats: 100 random grouped splits at 8--320 labeled target trajectories.",
        "- Operating point: trajectory-level 10% false-positive rate among final-correct trajectories.",
        "- `oracle_full_target` is an analysis upper bound and is not a deployable result.",
        "",
        "## Key Results",
        "",
        "| Monitor | Labels | Calibration | FPR | Abs. FPR error | Pre-event recall | Mean correct calib. |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    selected_sizes = (64, 128, 320)
    selected_rules = ("source_fixed", "unlabeled_rate_match", "target_empirical", "target_conformal")
    for method in ("teacher", "observed", "direct_dynamic", "fixed_dynamic"):
        for size in selected_sizes:
            for rule in selected_rules:
                row = summary_row(payload, method, rule, size)
                lines.append(
                    f"| {METHOD_LABELS[method]} | {size} | `{rule}` | "
                    f"{row['actual_fpr']['mean']:.3f} | {row['abs_fpr_error']['mean']:.3f} | "
                    f"{row['pre_event_recall']['mean']:.3f} | {row['mean_calibration_correct']:.1f} |"
                )
    lines.extend([
        "",
        "## Probability Calibration",
        "",
        "The target-intercept method keeps the source slope and updates one scalar intercept from target labels. It is strictly monotone and therefore leaves AUROC unchanged.",
        "",
        "| Monitor | Target labels | Calibration | Brier | NLL | ECE-10 | AUROC |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for method in ("teacher", "observed", "direct_dynamic", "fixed_dynamic"):
        for size in (128, 320):
            for rule in ("raw_score", "source_platt", "target_intercept"):
                row = next(
                    item for item in payload["probability_summary"]
                    if item["method"] == method
                    and item["calibration_method"] == rule
                    and item["n_calibration"] == size
                )
                lines.append(
                    f"| {METHOD_LABELS[method]} | {size} | `{rule}` | "
                    f"{row['brier']['mean']:.3f} | {row['nll']['mean']:.3f} | "
                    f"{row['ece_10']['mean']:.3f} | {row['auroc']['mean']:.3f} |"
                )
    lines.extend([
        "",
        "## Paired Split Stability at 320 Labels",
        "",
        "| Monitor vs dense teacher | Rule | Recall delta | FPR delta | Dominance rate |",
        "|---|---|---:|---:|---:|",
    ])
    for method in ("observed", "direct_dynamic", "fixed_dynamic"):
        for rule in ("target_empirical", "target_conformal"):
            row = next(
                item for item in payload["paired_comparisons"]
                if item["method"] == method
                and item["calibration_method"] == rule
                and item["n_calibration"] == 320
            )
            lines.append(
                f"| {METHOD_LABELS[method]} | `{rule}` | {row['recall_delta_mean']:+.3f} | "
                f"{row['fpr_delta_mean']:+.3f} | {row['dominance_rate']:.2f} |"
            )
    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- Monotone recalibration cannot improve ranking AUROC; it only transports the risk scale and operating threshold.",
        "- HotpotQA contains 66 correct trajectories out of 640. Consequently, 64 randomly labeled trajectories provide only about 6--7 correct calibration examples on average.",
        "- Split-conformal control becomes nontrivial at 10% FPR only after at least nine correct calibration examples; tighter high-confidence guarantees require substantially more.",
        "- Dense-versus-sparse conclusions should therefore use both ranking retention from FPE8c and threshold recovery from FPE9.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(payload: dict) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {
        "source_fixed": ("#777777", "--"),
        "unlabeled_rate_match": ("#1f77b4", "-"),
        "target_empirical": ("#2ca02c", "-"),
        "target_conformal": ("#d62728", "-"),
    }
    method = "direct_dynamic"
    for rule, (color, linestyle) in styles.items():
        rows = sorted(
            (row for row in payload["threshold_summary"] if row["method"] == method and row["calibration_method"] == rule),
            key=lambda row: row["n_calibration"],
        )
        x = [row["n_calibration"] for row in rows]
        axes[0].plot(x, [row["actual_fpr"]["mean"] for row in rows], marker="o", color=color, linestyle=linestyle, label=rule)
        axes[1].plot(x, [row["pre_event_recall"]["mean"] for row in rows], marker="o", color=color, linestyle=linestyle, label=rule)
    axes[0].axhline(TARGET_FPR, color="black", linewidth=1, linestyle=":")
    axes[0].set_ylabel("Held-out target FPR")
    axes[1].set_ylabel("Held-out pre-event recall")
    for axis in axes:
        axis.set_xlabel("Labeled target trajectories")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("FPE9: Sparse dynamic K=8 cross-domain calibration")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_key_results(payload: dict) -> None:
    for size in (64, 128, 320):
        print(f"n={size}")
        for method in ("teacher", "observed", "direct_dynamic", "fixed_dynamic"):
            pieces = []
            for rule in ("source_fixed", "unlabeled_rate_match", "target_empirical", "target_conformal"):
                row = summary_row(payload, method, rule, size)
                pieces.append(
                    f"{rule}:FPR={row['actual_fpr']['mean']:.3f},R={row['pre_event_recall']['mean']:.3f}"
                )
            print(f"  {method:<16} " + " | ".join(pieces))


if __name__ == "__main__":
    run()
