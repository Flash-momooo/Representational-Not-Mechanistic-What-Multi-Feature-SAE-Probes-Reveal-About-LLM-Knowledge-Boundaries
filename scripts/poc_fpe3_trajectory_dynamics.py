"""FPE3: question-gauged discrete trajectory energy without sklearn probes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 29, 42)
STAGES = tuple(range(6))
RIDGE = 0.10
MAX_ITER = 100
TOLERANCE = 1e-7


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


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels == 1].sum() / n_pos)


def available_methods(stage: int) -> tuple[str, ...]:
    methods = ["confidence", "token_prefix", "position_l18"]
    if stage >= 1:
        methods.extend(("velocity_l18", "kinetic_fusion"))
    if stage >= 2:
        methods.append("acceleration_fusion")
    return tuple(methods)


def feature_blocks(states: dict[str, np.ndarray], stage: int, method: str) -> list[np.ndarray]:
    confidence = states[f"confidence_T{stage}"].astype(np.float32)
    position = states[f"raw_T{stage}_L18"].astype(np.float32)
    if method == "confidence":
        return [confidence]
    if method == "token_prefix":
        return [states[f"token_prefix_T{stage}"].astype(np.float32)]
    if method == "position_l18":
        return [position]
    previous = states[f"raw_T{stage - 1}_L18"].astype(np.float32)
    velocity = position - previous
    if method == "velocity_l18":
        return [velocity]
    susceptibility = 4.0 * confidence[:, :1] * (1.0 - confidence[:, :1])
    scaled_velocity = susceptibility * velocity
    if method == "kinetic_fusion":
        return [scaled_velocity, confidence]
    if method == "acceleration_fusion":
        pre_previous = states[f"raw_T{stage - 2}_L18"].astype(np.float32)
        previous_velocity = previous - pre_previous
        acceleration = velocity - previous_velocity
        return [scaled_velocity, susceptibility * acceleration, confidence]
    raise ValueError(method)


def discordant_questions(
    question_ids: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    output = []
    for question in np.unique(question_ids[valid]):
        local = valid & (question_ids == question)
        if len(np.unique(labels[local])) == 2:
            output.append(question)
    return np.asarray(output, dtype=str)


def fit_scalers(blocks: list[np.ndarray], train_rows: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    scalers = []
    for block in blocks:
        train = block[train_rows]
        mean = train.mean(axis=0, keepdims=True)
        rms = np.maximum(
            np.sqrt(np.mean((train - mean) ** 2, axis=0, keepdims=True)), 1e-5
        )
        scalers.append((mean.astype(np.float32), rms.astype(np.float32)))
    return scalers


def transform(
    blocks: list[np.ndarray], scalers: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    output = []
    for block, (mean, rms) in zip(blocks, scalers):
        output.append((block - mean) / rms / np.sqrt(block.shape[1]))
    return np.concatenate(output, axis=1).astype(np.float32)


def question_weighted_differences(
    features: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    differences = []
    weights = []
    questions = discordant_questions(question_ids, labels, valid)
    for question in questions:
        rows = np.flatnonzero(valid & (question_ids == question))
        wrong = rows[labels[rows] == 1]
        correct = rows[labels[rows] == 0]
        local = (
            features[wrong, None, :] - features[correct, :][None, :, :]
        ).reshape(-1, features.shape[1])
        differences.append(local)
        weights.append(np.full(len(local), 1.0 / len(local), dtype=np.float32))
    if not differences:
        return (
            np.zeros((0, features.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            0,
        )
    return np.vstack(differences), np.concatenate(weights), len(questions)


def solve_energy(
    differences: np.ndarray, weights: np.ndarray, device: str
) -> tuple[np.ndarray, dict]:
    if len(differences) == 0:
        raise ValueError("No discordant training questions")
    x = torch.as_tensor(differences, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    normalizer = sample_weights.sum()
    moment = x.T @ sample_weights / normalizer
    initial_norm = torch.linalg.vector_norm(moment)
    if float(initial_norm) <= 1e-20:
        return np.zeros(x.shape[1], dtype=np.float32), {
            "iterations": 0, "relative_residual": 0.0, "weight_norm": 0.0,
        }

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return RIDGE * vector + x.T @ (sample_weights * (x @ vector)) / normalizer

    solution = torch.zeros_like(moment)
    residual = moment.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    relative_residual = 1.0
    iterations = 0
    for iterations in range(1, MAX_ITER + 1):
        response = matvec(direction)
        step = residual_sq / torch.dot(direction, response).clamp_min(1e-20)
        solution = solution + step * direction
        residual = residual - step * response
        new_residual_sq = torch.dot(residual, residual)
        relative_residual = float(torch.sqrt(new_residual_sq) / initial_norm)
        if relative_residual <= TOLERANCE:
            break
        direction = residual + (new_residual_sq / residual_sq.clamp_min(1e-20)) * direction
        residual_sq = new_residual_sq
    diagnostics = {
        "iterations": iterations,
        "relative_residual": relative_residual,
        "weight_norm": float(torch.linalg.vector_norm(solution)),
    }
    return solution.cpu().numpy(), diagnostics


def split_folds(question_ids: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(question_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {question: index % 5 for index, question in enumerate(unique)}
    return [
        (
            np.asarray([mapping[question] != fold for question in question_ids]),
            np.asarray([mapping[question] == fold for question in question_ids]),
        )
        for fold in range(5)
    ]


def score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    question_ids: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict, dict[str, float]]:
    scored = valid & np.isfinite(scores)
    local = {}
    for question in np.unique(question_ids[scored]):
        mask = scored & (question_ids == question)
        if len(np.unique(labels[mask])) == 2:
            local[question] = auroc(labels[mask], scores[mask])
    return {
        "population_auroc": auroc(labels[scored], scores[scored]),
        "population_auprc": auprc(labels[scored], scores[scored]),
        "within_question_auroc_macro": float(np.mean(list(local.values()))),
        "n_rows": int(scored.sum()),
        "n_discordant_questions": len(local),
    }, local


def bootstrap(values: np.ndarray, seed: int, reference: float) -> dict:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "reference": reference,
        "probability_at_or_below_reference": float(np.mean(samples <= reference)),
        "n_questions": len(values),
    }


def run(cache_path: Path, output_path: Path, device: str) -> dict:
    cache = np.load(cache_path, allow_pickle=True)
    states = {key: cache[key] for key in cache.files}
    labels = states["labels"].astype(np.int8)
    question_ids = states["question_ids"].astype(str)
    rows = []
    local_store: dict[tuple[int, str, int], dict[str, float]] = {}
    diagnostics = []

    for stage in STAGES:
        valid = states[f"valid_T{stage}"].astype(bool)
        methods = available_methods(stage)
        for seed in SEEDS:
            predictions = {
                method: np.full(len(labels), np.nan, dtype=np.float32)
                for method in methods
            }
            for fold, (train_question_rows, test_question_rows) in enumerate(
                split_folds(question_ids, seed)
            ):
                train_valid = valid & train_question_rows
                train_questions = discordant_questions(question_ids, labels, train_valid)
                train_rows = train_valid & np.isin(question_ids, train_questions)
                test_rows = valid & test_question_rows
                for method in methods:
                    blocks = feature_blocks(states, stage, method)
                    scalers = fit_scalers(blocks, train_rows)
                    features = transform(blocks, scalers)
                    differences, weights, n_train_questions = question_weighted_differences(
                        features, labels, question_ids, train_rows
                    )
                    energy, fit_info = solve_energy(differences, weights, device)
                    predictions[method][test_rows] = features[test_rows] @ energy
                    diagnostics.append({
                        "stage": f"T{stage}", "method": method, "seed": seed,
                        "fold": fold, "n_train_questions": n_train_questions,
                        **fit_info,
                    })
            for method in methods:
                metrics, local = score_metrics(
                    labels, predictions[method], question_ids, valid
                )
                local_store[(stage, method, seed)] = local
                rows.append({
                    "stage": f"T{stage}", "method": method, "seed": seed, **metrics
                })

    summary = []
    for stage in STAGES:
        for method in available_methods(stage):
            selected = [
                row for row in rows
                if row["stage"] == f"T{stage}" and row["method"] == method
            ]
            summary.append({
                "stage": f"T{stage}", "method": method,
                "within_question_auroc_macro_mean": float(np.mean([
                    row["within_question_auroc_macro"] for row in selected
                ])),
                "within_question_auroc_macro_std": float(np.std([
                    row["within_question_auroc_macro"] for row in selected
                ], ddof=1)),
                "population_auroc_mean": float(np.mean([
                    row["population_auroc"] for row in selected
                ])),
                "population_auprc_mean": float(np.mean([
                    row["population_auprc"] for row in selected
                ])),
                "n_rows": selected[0]["n_rows"],
                "n_discordant_questions": selected[0]["n_discordant_questions"],
            })

    comparisons = []
    for stage in STAGES:
        for method in available_methods(stage):
            common = sorted(set.intersection(*[
                set(local_store[(stage, method, seed)]) for seed in SEEDS
            ]))
            values = np.asarray([
                np.mean([local_store[(stage, method, seed)][question] for seed in SEEDS])
                for question in common
            ])
            comparisons.append({
                "contrast": f"T{stage} {method} versus chance",
                "bootstrap": bootstrap(values, 2026071800 + stage, 0.5),
            })
        if stage >= 1:
            for baseline_index, baseline in enumerate(("confidence", "token_prefix"), 1):
                common = sorted(set.intersection(*[
                    set(local_store[(stage, method, seed)])
                    for method in ("kinetic_fusion", baseline) for seed in SEEDS
                ]))
                differences = np.asarray([
                    np.mean([
                        local_store[(stage, "kinetic_fusion", seed)][question]
                        - local_store[(stage, baseline, seed)][question]
                        for seed in SEEDS
                    ]) for question in common
                ])
                comparisons.append({
                    "contrast": f"T{stage} kinetic_fusion minus {baseline}",
                    "bootstrap": bootstrap(
                        differences, 2026071850 + stage * 2 + baseline_index, 0.0
                    ),
                })
            for comparison_index, (left, right) in enumerate((
                ("velocity_l18", "position_l18"),
                ("kinetic_fusion", "velocity_l18"),
            ), 1):
                common = sorted(set.intersection(*[
                    set(local_store[(stage, method, seed)])
                    for method in (left, right) for seed in SEEDS
                ]))
                differences = np.asarray([
                    np.mean([
                        local_store[(stage, left, seed)][question]
                        - local_store[(stage, right, seed)][question]
                        for seed in SEEDS
                    ]) for question in common
                ])
                comparisons.append({
                    "contrast": f"T{stage} {left} minus {right}",
                    "bootstrap": bootstrap(
                        differences, 2026071950 + stage * 2 + comparison_index, 0.0
                    ),
                })
        if stage >= 2:
            common = sorted(set.intersection(*[
                set(local_store[(stage, method, seed)])
                for method in ("acceleration_fusion", "kinetic_fusion")
                for seed in SEEDS
            ]))
            differences = np.asarray([
                np.mean([
                    local_store[(stage, "acceleration_fusion", seed)][question]
                    - local_store[(stage, "kinetic_fusion", seed)][question]
                    for seed in SEEDS
                ]) for question in common
            ])
            comparisons.append({
                "contrast": f"T{stage} acceleration_fusion minus kinetic_fusion",
                "bootstrap": bootstrap(differences, 2026071900 + stage, 0.0),
            })

    payload = {
        "experiment": "FPE3 first-principles trajectory dynamics",
        "target": "eventual answer error from prefix through token t",
        "constants": {"ridge": RIDGE, "max_iter": MAX_ITER, "tolerance": TOLERANCE},
        "n_rows": len(labels),
        "n_questions": len(np.unique(question_ids)),
        "summary": summary,
        "comparisons": comparisons,
        "local_aurocs": {
            f"T{stage}|{method}|{seed}": local
            for (stage, method, seed), local in local_store.items()
        },
        "rows": rows,
        "fit_diagnostics": diagnostics,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output_path}")
    for row in summary:
        print(
            f"{row['stage']} {row['method']:<21} "
            f"within={row['within_question_auroc_macro_mean']:.4f} "
            f"pop={row['population_auroc_mean']:.4f}"
        )
    for comparison in comparisons:
        if "kinetic_fusion minus" in comparison["contrast"]:
            print(comparison)
    return payload


def main() -> None:
    cache = ROOT / "outputs" / "cache" / "v37_2wiki_trajectory_states.npz"
    output = ROOT / "outputs" / "poc_fpe3_trajectory_dynamics_results.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(cache, output, device)


if __name__ == "__main__":
    main()
