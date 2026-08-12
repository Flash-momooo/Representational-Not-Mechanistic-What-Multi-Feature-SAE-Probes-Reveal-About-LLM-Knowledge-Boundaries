"""V43: distinguish question-level T0 risk from trajectory-level T0 outcome.

This is a cache-only retrospective audit. It uses one shared pre-generation
state per question to predict the empirical error rate across eight sampled
answers. The same question-level scores are then copied to the eight answers:
their strict within-question AUROC is necessarily 0.5 whenever a question has
both correct and incorrect samples.

The script reuses the FPE4 analysis family without target-side parameter
selection: five grouped folds, seeds 13/29/42, Beta(1,1) target smoothing, and
ridge 0.10. Because the caches predate this audit, the resulting analysis is
retrospective rather than a new prospective confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 29, 42)
RIDGE = 0.10
MAX_ITER = 100
TOLERANCE = 1e-7
METHODS = ("global", "confidence_T0", "raw_T0", "fused_T0")


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    return float(np.sum(left * right) / denominator) if denominator > 0 else 0.0


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def load_questions(cache_path: Path) -> tuple[dict[str, np.ndarray], float]:
    states = np.load(cache_path, allow_pickle=True)
    required = ("labels", "question_ids", "confidence_T0", "raw_T0")
    missing = [key for key in required if key not in states.files]
    if missing:
        raise KeyError(f"cache missing {missing}: {cache_path}")

    question_ids = states["question_ids"].astype(str)
    labels = states["labels"].astype(np.int8)
    rows = {
        "question_id": [],
        "errors": [],
        "trials": [],
        "observed_rate": [],
        "posterior_rate": [],
        "confidence": [],
        "raw": [],
    }
    maxima = []
    for question_id in np.unique(question_ids):
        mask = question_ids == question_id
        local_labels = labels[mask]
        local_raw = states["raw_T0"][mask].astype(np.float32)
        maxima.append(float(np.max(np.var(local_raw, axis=0))))
        errors = int(local_labels.sum())
        trials = int(mask.sum())
        rows["question_id"].append(question_id)
        rows["errors"].append(errors)
        rows["trials"].append(trials)
        rows["observed_rate"].append(errors / trials)
        rows["posterior_rate"].append((errors + 1) / (trials + 2))
        rows["confidence"].append(states["confidence_T0"][mask][0].astype(np.float32))
        rows["raw"].append(local_raw[0])

    output = {
        key: np.asarray(value, dtype=str if key == "question_id" else np.float32)
        for key, value in rows.items()
    }
    return output, float(np.max(maxima))


def make_folds(question_ids: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(question_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_map = {question_id: index % 5 for index, question_id in enumerate(unique)}
    return [
        (
            np.asarray([fold_map[question_id] != fold for question_id in question_ids]),
            np.asarray([fold_map[question_id] == fold for question_id in question_ids]),
        )
        for fold in range(5)
    ]


def standardize(blocks: list[np.ndarray], train: np.ndarray) -> np.ndarray:
    scaled = []
    for block in blocks:
        mean = block[train].mean(axis=0, keepdims=True)
        rms = np.maximum(
            np.sqrt(np.mean((block[train] - mean) ** 2, axis=0, keepdims=True)),
            1e-5,
        )
        scaled.append((block - mean) / rms / np.sqrt(block.shape[1]))
    return np.concatenate(scaled, axis=1).astype(np.float32)


def ridge_predict(
    blocks: list[np.ndarray],
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    device: str,
) -> np.ndarray:
    features = standardize(blocks, train)
    target = np.clip(target, 1e-5, 1.0 - 1e-5)
    target_logit = np.log(target / (1.0 - target))
    intercept = float(target_logit[train].mean())
    x = torch.as_tensor(features[train], dtype=torch.float32, device=device)
    y = torch.as_tensor(target_logit[train] - intercept, dtype=torch.float32, device=device)
    moment = x.T @ y / len(x)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return RIDGE * vector + x.T @ (x @ vector) / len(x)

    weights = torch.zeros_like(moment)
    residual = moment.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial_norm = torch.linalg.vector_norm(residual)
    if float(initial_norm) <= 1e-20:
        return np.full(int(test.sum()), sigmoid(np.asarray([intercept]))[0])
    for _ in range(MAX_ITER):
        response = matvec(direction)
        step = residual_sq / torch.dot(direction, response).clamp_min(1e-20)
        weights = weights + step * direction
        residual = residual - step * response
        new_residual_sq = torch.dot(residual, residual)
        if float(torch.sqrt(new_residual_sq) / initial_norm) <= TOLERANCE:
            break
        direction = residual + (new_residual_sq / residual_sq.clamp_min(1e-20)) * direction
        residual_sq = new_residual_sq
    return sigmoid(intercept + features[test] @ weights.detach().cpu().numpy())


def metric_row(errors: np.ndarray, trials: np.ndarray, prediction: np.ndarray) -> tuple[dict, np.ndarray]:
    prediction = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    observed = errors / trials
    nll_sum = -(errors * np.log(prediction) + (trials - errors) * np.log(1.0 - prediction))
    return {
        "binomial_nll_per_completion": float(nll_sum.sum() / trials.sum()),
        "mse_observed_rate": float(np.mean((prediction - observed) ** 2)),
        "pearson": pearson(prediction, observed),
        "spearman": spearman(prediction, observed),
    }, nll_sum / trials


def bootstrap_delta(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(samples, (0.025, 0.975))],
    }


def run(cache_path: Path, output_path: Path) -> dict:
    data, max_variance = load_questions(cache_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions_by_method: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    loss_by_method: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    rows = []

    for seed in SEEDS:
        predictions = {
            method: np.full(len(data["question_id"]), np.nan, dtype=np.float64)
            for method in METHODS
        }
        for train, test in make_folds(data["question_id"], seed):
            predictions["global"][test] = float(data["posterior_rate"][train].mean())
            feature_sets = {
                "confidence_T0": [data["confidence"]],
                "raw_T0": [data["raw"]],
                "fused_T0": [data["raw"], data["confidence"]],
            }
            for method, blocks in feature_sets.items():
                predictions[method][test] = ridge_predict(
                    blocks, data["posterior_rate"], train, test, device
                )
        for method in METHODS:
            metrics, losses = metric_row(data["errors"], data["trials"], predictions[method])
            predictions_by_method[method].append(predictions[method])
            loss_by_method[method].append(losses)
            rows.append({"method": method, "seed": seed, **metrics})

    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            **{
                f"{metric}_mean": float(np.mean([row[metric] for row in selected]))
                for metric in ("binomial_nll_per_completion", "mse_observed_rate", "pearson", "spearman")
            },
            "strict_within_question_trajectory_auroc": 0.5,
        })
    summary.sort(key=lambda row: row["binomial_nll_per_completion_mean"])

    mean_loss = {method: np.mean(loss_by_method[method], axis=0) for method in METHODS}
    comparisons = [
        {
            "contrast": "fused_T0 minus global binomial NLL",
            "bootstrap": bootstrap_delta(mean_loss["fused_T0"] - mean_loss["global"], 2026080401),
        },
        {
            "contrast": "fused_T0 minus confidence_T0 binomial NLL",
            "bootstrap": bootstrap_delta(mean_loss["fused_T0"] - mean_loss["confidence_T0"], 2026080402),
        },
    ]
    payload = {
        "experiment": "V43 T0 target separation retrospective audit",
        "status": "cache-only retrospective reanalysis; not a prospective confirmation",
        "cache": str(cache_path),
        "model_condition": cache_path.stem,
        "n_questions": int(len(data["question_id"])),
        "samples_per_question": int(data["trials"][0]),
        "max_within_question_T0_raw_feature_variance": max_variance,
        "question_target": "Beta(1,1)-smoothed empirical error rate across eight stochastic completions",
        "trajectory_target": "individual completion error within the same question",
        "analysis_protocol": "Retrospective reuse of FPE4 settings: 5 question-grouped folds, seeds 13/29/42, ridge 0.10; no target-side parameter selection",
        "summary": summary,
        "comparisons": comparisons,
        "rows": rows,
        "interpretation": {
            "question_level": "A shared T0 state may rank questions by expected completion error.",
            "trajectory_level": "A deterministic T0-only score is identical for all completions of one question and therefore has strict within-question AUROC 0.5.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(ROOT / args.cache, ROOT / args.output)


if __name__ == "__main__":
    main()
