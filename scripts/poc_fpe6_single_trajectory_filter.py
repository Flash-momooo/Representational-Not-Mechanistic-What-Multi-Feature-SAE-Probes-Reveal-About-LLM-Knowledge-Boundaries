"""FPE6: single-trajectory R0 + Rt + EC risk filter without sklearn."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from scripts.poc_fpe3_trajectory_dynamics import (
    auprc,
    auroc,
    fit_scalers,
    question_weighted_differences,
    solve_energy,
    transform,
)
from scripts.poc_fpe5_first_divergence_hazard import load_event_table

RIDGE = 0.10


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def apply_scalers(blocks, scalers):
    return np.concatenate([
        (block.astype(np.float32) - mean) / rms / np.sqrt(block.shape[1])
        for block, (mean, rms) in zip(blocks, scalers)
    ], axis=1).astype(np.float32)


def fit_logistic(features: np.ndarray, labels: np.ndarray, ridge: float = 0.10):
    x = np.column_stack((np.ones(len(features)), features)).astype(np.float64)
    y = labels.astype(np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(100):
        probabilities = sigmoid(x @ weights)
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-5)
        gradient = x.T @ (probabilities - y) / len(y)
        gradient[1:] += ridge * weights[1:]
        hessian = (x.T * curvature) @ x / len(y)
        hessian[1:, 1:] += ridge * np.eye(x.shape[1] - 1)
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return weights


def logistic_predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    x = np.column_stack((np.ones(len(features)), features)).astype(np.float64)
    return sigmoid(x @ weights)


def fit_pair_model(blocks, labels, question_ids, train_valid, device):
    scalers = fit_scalers(blocks, train_valid)
    features = transform(blocks, scalers)
    differences, weights, n_questions = question_weighted_differences(
        features, labels, question_ids, train_valid
    )
    direction, diagnostics = solve_energy(differences, weights, device)
    raw_scores = features[train_valid] @ direction
    calibrator = fit_logistic(raw_scores[:, None], labels[train_valid], ridge=0.02)
    return {
        "scalers": scalers, "direction": direction, "calibrator": calibrator,
        "n_discordant_questions": n_questions, "diagnostics": diagnostics,
    }


def predict_pair_model(model, blocks):
    features = apply_scalers(blocks, model["scalers"])
    scores = features @ model["direction"]
    return logistic_predict(model["calibrator"], scores[:, None])


def solve_ridge(features: np.ndarray, targets: np.ndarray, device: str):
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    intercept = float(y.mean())
    moment = x.T @ (y - intercept) / len(x)

    def matvec(vector):
        return RIDGE * vector + x.T @ (x @ vector) / len(x)

    solution = torch.zeros_like(moment)
    residual = moment.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial = torch.linalg.vector_norm(residual).clamp_min(1e-20)
    for _ in range(100):
        response = matvec(direction)
        step = residual_sq / torch.dot(direction, response).clamp_min(1e-20)
        solution += step * direction
        residual -= step * response
        new_sq = torch.dot(residual, residual)
        if float(torch.sqrt(new_sq) / initial) <= 1e-7:
            break
        direction = residual + (new_sq / residual_sq.clamp_min(1e-20)) * direction
        residual_sq = new_sq
    return intercept, solution.cpu().numpy()


def aggregate_r0(states, labels, valid):
    question_ids = states["question_ids"].astype(str)
    output = {"rows": [], "rates": [], "raw": [], "confidence": []}
    for question in np.unique(question_ids[valid]):
        indices = np.flatnonzero(valid & (question_ids == question))
        errors = int(labels[indices].sum())
        output["rows"].append(indices[0])
        output["rates"].append((errors + 1.0) / (len(indices) + 2.0))
        output["raw"].append(states["raw_T0_L18"][indices[0]].astype(np.float32))
        output["confidence"].append(states["confidence_T0"][indices[0]].astype(np.float32))
    return {key: np.asarray(value) for key, value in output.items()}


def fit_r0(states, labels, train_valid, device):
    data = aggregate_r0(states, labels, train_valid)
    blocks = [data["raw"], data["confidence"]]
    local = np.ones(len(data["rates"]), dtype=bool)
    scalers = fit_scalers(blocks, local)
    features = transform(blocks, scalers)
    intercept, weights = solve_ridge(features, logit(data["rates"]), device)
    return {"scalers": scalers, "intercept": intercept, "weights": weights}


def predict_r0(model, states):
    blocks = [
        states["raw_T0_L18"].astype(np.float32),
        states["confidence_T0"].astype(np.float32),
    ]
    features = apply_scalers(blocks, model["scalers"])
    return sigmoid(model["intercept"] + features @ model["weights"])


def phase_blocks(states, phase):
    if phase == "rt_dense":
        return [
            states["raw_T1_L18"].astype(np.float32)
            - states["raw_T0_L18"].astype(np.float32),
            states["token_prefix_T1"].astype(np.float32),
            states["confidence_T1"].astype(np.float32),
        ]
    if phase == "rt_sae":
        return [
            states["sae_T1_L18"].astype(np.float32)
            - states["sae_T0_L18"].astype(np.float32),
            states["token_prefix_T1"].astype(np.float32),
            states["confidence_T1"].astype(np.float32),
        ]
    if phase == "ec_dense":
        return [
            states["raw_C_L18"].astype(np.float32)
            - states["raw_T0_L18"].astype(np.float32),
            states["confidence_C"].astype(np.float32),
        ]
    if phase == "ec_sae":
        return [
            states["sae_C_L18"].astype(np.float32)
            - states["sae_T0_L18"].astype(np.float32),
            states["confidence_C"].astype(np.float32),
        ]
    if phase == "confidence":
        return [states["confidence_C"].astype(np.float32)]
    if phase == "token":
        return [states["token_prefix_C"].astype(np.float32)]
    raise ValueError(phase)


def pair_folds(pair_ids, seed=42):
    unique = np.unique(pair_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {pair_id: index % 5 for index, pair_id in enumerate(unique)}
    for fold in range(5):
        yield (
            np.asarray([mapping[pair_id] != fold for pair_id in pair_ids]),
            np.asarray([mapping[pair_id] == fold for pair_id in pair_ids]),
        )


def train_phases(states, labels, event_table, train_rows, device):
    question_ids = states["question_ids"].astype(str)
    models = {"r0": fit_r0(states, labels, train_rows, device)}
    rt_valid = train_rows & event_table["risk"][1]
    for phase in ("rt_dense", "rt_sae"):
        models[phase] = fit_pair_model(
            phase_blocks(states, phase), event_table["event"][1],
            question_ids, rt_valid, device,
        )
    for phase in ("ec_dense", "ec_sae", "confidence", "token"):
        models[phase] = fit_pair_model(
            phase_blocks(states, phase), labels, question_ids, train_rows, device
        )
    return models


def predict_phases(models, states):
    output = {"r0": predict_r0(models["r0"], states)}
    t1_valid = states["valid_T1"].astype(bool)
    for phase in ("rt_dense", "rt_sae"):
        probabilities = np.ones(len(t1_valid), dtype=np.float64)
        predicted = predict_pair_model(models[phase], phase_blocks(states, phase))
        probabilities[t1_valid] = predicted[t1_valid]
        output[phase] = probabilities
    for phase in ("ec_dense", "ec_sae", "confidence", "token"):
        output[phase] = predict_pair_model(models[phase], phase_blocks(states, phase))
    return output


def fit_combiners(oof, labels):
    definitions = {
        "r0": ("r0",),
        "rt_sae": ("rt_sae",),
        "ec_sae": ("ec_sae",),
        "r0_rt": ("r0", "rt_sae"),
        "r0_ec": ("r0", "ec_sae"),
        "rt_ec": ("rt_sae", "ec_sae"),
        "three_phase_sae": ("r0", "rt_sae", "ec_sae"),
        "three_phase_dense": ("r0", "rt_dense", "ec_dense"),
    }
    combiners = {}
    for name, phases in definitions.items():
        features = np.column_stack([logit(oof[phase]) for phase in phases])
        combiners[name] = {
            "phases": phases,
            "weights": fit_logistic(features, labels, ridge=0.10),
        }
    return combiners


def apply_combiners(combiners, phases):
    output = {}
    for name, model in combiners.items():
        features = np.column_stack([logit(phases[phase]) for phase in model["phases"]])
        output[name] = logistic_predict(model["weights"], features)
    output["confidence"] = phases["confidence"]
    output["token"] = phases["token"]
    return output


def ece(labels, probabilities, bins=10):
    total = 0.0
    for lower in np.linspace(0.0, 0.9, bins):
        upper = lower + 0.1
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if selected.any():
            total += selected.mean() * abs(
                probabilities[selected].mean() - labels[selected].mean()
            )
    return float(total)


def metrics(labels, probabilities, question_ids):
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    local = []
    for question in np.unique(question_ids):
        selected = question_ids == question
        if len(np.unique(labels[selected])) == 2:
            local.append(auroc(labels[selected], probabilities[selected]))
    return {
        "auroc": auroc(labels, probabilities),
        "auprc": auprc(labels, probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(np.mean(-labels * np.log(probabilities) - (1-labels) * np.log(1-probabilities))),
        "ece10": ece(labels, probabilities),
        "within_question_auroc_macro": float(np.mean(local)),
        "n_discordant_questions": len(local),
    }


def bootstrap_delta(labels, left, right, question_ids, seed=20260715):
    values = []
    for question in np.unique(question_ids):
        selected = question_ids == question
        if len(np.unique(labels[selected])) == 2:
            values.append(
                auroc(labels[selected], left[selected])
                - auroc(labels[selected], right[selected])
            )
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(sampled, (0.025, 0.975))],
        "n_questions": len(values),
    }


def warning_audit(train_labels, train_scores, test_labels, test_scores, target_table):
    thresholds = {
        phase: float(np.quantile(scores[train_labels == 0], 0.90))
        for phase, scores in train_scores.items()
    }
    onset = np.full(len(test_labels), np.nan)
    for stage in range(6):
        selected = target_table["risk"][stage] & (target_table["event"][stage] == 1)
        onset[np.isnan(onset) & selected] = stage + 1
    wrong_observed = (test_labels == 1) & np.isfinite(onset)
    rows = []
    for phase, stage in (("r0", 0), ("rt_sae", 1), ("rt_dense", 1)):
        warned = test_scores[phase] >= thresholds[phase]
        detected = wrong_observed & warned & (onset > stage)
        rows.append({
            "phase": phase, "threshold_train_correct_fpr_0_10": thresholds[phase],
            "target_correct_fpr": float(np.mean(warned[test_labels == 0])),
            "observed_wrong_trajectories": int(wrong_observed.sum()),
            "detected_before_event": int(detected.sum()),
            "pre_event_detection_rate": float(detected.sum() / max(wrong_observed.sum(), 1)),
            "mean_token_lead_when_detected": (
                float(np.mean(onset[detected] - stage)) if detected.any() else None
            ),
        })
    return rows


def serializable_model(models, combiners, path):
    arrays = {}
    for name, model in models.items():
        if name == "r0":
            arrays[f"{name}_weights"] = model["weights"]
            arrays[f"{name}_intercept"] = np.asarray([model["intercept"]])
        else:
            arrays[f"{name}_direction"] = model["direction"]
            arrays[f"{name}_calibrator"] = model["calibrator"]
        for index, (mean, rms) in enumerate(model["scalers"]):
            arrays[f"{name}_mean_{index}"] = mean
            arrays[f"{name}_rms_{index}"] = rms
    for name, model in combiners.items():
        arrays[f"combiner_{name}"] = model["weights"]
    np.savez_compressed(path, **arrays)


def run():
    config = yaml.safe_load((ROOT / "configs" / "fpe5_l18.yaml").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], local_files_only=True
    )
    source_data = ROOT / "data" / "fpe5c2_policy_full_trajectory.jsonl"
    source_cache = ROOT / "outputs" / "cache" / "fpe5c2_policy_full_trajectory_states.npz"
    target_data = ROOT / "data" / "fpe5c1_hotpot_trajectory.jsonl"
    target_cache = ROOT / "outputs" / "cache" / "fpe5c1_hotpot_trajectory_states.npz"
    _, source, source_events = load_event_table(tokenizer, source_data, source_cache)
    _, target, target_events = load_event_table(tokenizer, target_data, target_cache)
    source_labels = source["labels"].astype(np.int8)
    target_labels = target["labels"].astype(np.int8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = {
        phase: np.full(len(source_labels), np.nan, dtype=np.float64)
        for phase in ("r0", "rt_dense", "rt_sae", "ec_dense", "ec_sae", "confidence", "token")
    }
    for train, test in pair_folds(source["pair_ids"].astype(str), seed=42):
        fold_models = train_phases(source, source_labels, source_events["table"], train, device)
        fold_predictions = predict_phases(fold_models, source)
        for phase in oof:
            oof[phase][test] = fold_predictions[phase][test]
    assert all(np.isfinite(values).all() for values in oof.values())
    combiners = fit_combiners(oof, source_labels)
    source_scores = apply_combiners(combiners, oof)

    final_models = train_phases(
        source, source_labels, source_events["table"],
        np.ones(len(source_labels), dtype=bool), device,
    )
    target_phases = predict_phases(final_models, target)
    target_scores = apply_combiners(combiners, target_phases)
    target_question_ids = target["question_ids"].astype(str)
    results = {
        name: metrics(target_labels, scores, target_question_ids)
        for name, scores in target_scores.items()
    }
    comparisons = {
        "three_phase_sae_minus_confidence": bootstrap_delta(
            target_labels, target_scores["three_phase_sae"],
            target_scores["confidence"], target_question_ids, 2026071501,
        ),
        "three_phase_sae_minus_token": bootstrap_delta(
            target_labels, target_scores["three_phase_sae"],
            target_scores["token"], target_question_ids, 2026071502,
        ),
        "three_phase_sae_minus_dense": bootstrap_delta(
            target_labels, target_scores["three_phase_sae"],
            target_scores["three_phase_dense"], target_question_ids, 2026071503,
        ),
        "three_phase_sae_minus_rt_sae": bootstrap_delta(
            target_labels, target_scores["three_phase_sae"],
            target_scores["rt_sae"], target_question_ids, 2026071504,
        ),
    }
    warnings = warning_audit(
        source_labels,
        {phase: oof[phase] for phase in ("r0", "rt_sae", "rt_dense")},
        target_labels,
        {phase: target_phases[phase] for phase in ("r0", "rt_sae", "rt_dense")},
        target_events["table"],
    )
    model_path = ROOT / "outputs" / "fpe6_single_trajectory_filter.npz"
    serializable_model(final_models, combiners, model_path)
    payload = {
        "experiment": "FPE6 single-trajectory three-phase filter",
        "training": "2Wiki policy-B trajectories",
        "untouched_test": "HotpotQA C1 trajectories",
        "inference_constraints": {
            "one_trajectory": True, "question_centering": False,
            "sibling_samples": False, "gold_probability": False,
            "future_state_for_early_phases": False,
        },
        "n_train": len(source_labels), "n_test": len(target_labels),
        "test_error_rate": float(target_labels.mean()),
        "results": results, "comparisons": comparisons,
        "warning_lead_time": warnings,
        "serialized_model": str(model_path),
    }
    output = ROOT / "outputs" / "poc_fpe6_single_trajectory_filter_results.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output}")
    for name, row in sorted(results.items(), key=lambda item: item[1]["auroc"], reverse=True):
        print(
            f"{name:<20} AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} "
            f"Brier={row['brier']:.4f} ECE={row['ece10']:.4f} "
            f"within={row['within_question_auroc_macro']:.4f}"
        )
    print(json.dumps(comparisons, indent=2))
    print(json.dumps(warnings, indent=2))
    return payload


if __name__ == "__main__":
    run()
