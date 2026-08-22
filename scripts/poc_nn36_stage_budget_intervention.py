"""NN36: stage-wise, budget-constrained intervention on frozen trajectories.

This is a retrospective cache experiment. T0 allocates candidate budget from
question-level risk; T1--T3 may stop realized prefixes. All fitting,
calibration, and evaluation questions are disjoint inside each outer fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_frozen_representation_controls import (  # noqa: E402
    canonicalize,
    load_condition,
)
from scripts.poc_fpe3_trajectory_dynamics import fit_scalers, transform  # noqa: E402
from scripts.poc_fpe6_single_trajectory_filter import (  # noqa: E402
    apply_scalers,
    fit_pair_model,
    logit,
    predict_pair_model,
    sigmoid,
    solve_ridge,
)
from scripts.poc_fpe7_observability_utility import group_fold_ids  # noqa: E402


STAGES = (1, 2, 3)
TARGET_FPRS = (0.05, 0.10, 0.20)
FIXED_BUDGETS = (1, 2, 4, 8)
N_FOLDS = 5
EPS = 1e-7


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def method_blocks(states: dict[str, np.ndarray], stage: int, method: str):
    confidence = np.concatenate(
        (
            states["confidence_T0"].astype(np.float32),
            states[f"confidence_T{stage}"].astype(np.float32),
        ),
        axis=1,
    )
    if method == "confidence":
        return [confidence]
    observed = [confidence, states[f"token_prefix_T{stage}"].astype(np.float32)]
    if method == "observed":
        return observed
    if method == "dense":
        return observed + [
            states[f"raw_T{stage}"].astype(np.float32)
            - states["raw_T0"].astype(np.float32)
        ]
    if method == "sae":
        return observed + [
            states[f"sae_T{stage}"].astype(np.float32)
            - states["sae_T0"].astype(np.float32)
        ]
    raise ValueError(method)


def fit_stage_readout(
    states: dict[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    stage: int,
    method: str,
    device: str,
):
    valid = train & states[f"valid_T{stage}"].astype(bool)
    return fit_pair_model(
        method_blocks(states, stage, method),
        labels,
        states["question_ids"].astype(str),
        valid,
        device,
    )


def predict_stage_readout(model, states, stage: int, method: str) -> np.ndarray:
    return predict_pair_model(model, method_blocks(states, stage, method))


def aggregate_t0(
    states: dict[str, np.ndarray], labels: np.ndarray, selected: np.ndarray
) -> dict[str, np.ndarray]:
    question_ids = states["question_ids"].astype(str)
    rows, rates, raw, confidence = [], [], [], []
    for question_id in np.unique(question_ids[selected]):
        local = np.flatnonzero(selected & (question_ids == question_id))
        rows.append(local[0])
        rates.append((float(labels[local].sum()) + 1.0) / (len(local) + 2.0))
        raw.append(states["raw_T0"][local[0]].astype(np.float32))
        confidence.append(states["confidence_T0"][local[0]].astype(np.float32))
    return {
        "rows": np.asarray(rows, dtype=np.int64),
        "rates": np.asarray(rates, dtype=np.float64),
        "raw": np.asarray(raw, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
    }


def fit_t0(states, labels, train, device: str, confidence_only: bool = False):
    data = aggregate_t0(states, labels, train)
    blocks = [data["confidence"]] if confidence_only else [data["raw"], data["confidence"]]
    local = np.ones(len(data["rates"]), dtype=bool)
    scalers = fit_scalers(blocks, local)
    features = transform(blocks, scalers)
    intercept, weights = solve_ridge(features, logit(data["rates"]), device)
    return {
        "confidence_only": confidence_only,
        "scalers": scalers,
        "intercept": intercept,
        "weights": weights,
    }


def predict_t0(model, states) -> np.ndarray:
    blocks = [states["confidence_T0"].astype(np.float32)]
    if not model["confidence_only"]:
        blocks.insert(0, states["raw_T0"].astype(np.float32))
    features = apply_scalers(blocks, model["scalers"])
    return sigmoid(model["intercept"] + features @ model["weights"])


def stable_uniform(*parts) -> float:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def higher_quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("inf")
    return float(np.quantile(values, q, method="higher"))


def budget_cutpoints(scores: np.ndarray, question_ids: np.ndarray, calibration: np.ndarray):
    values = []
    for question_id in np.unique(question_ids[calibration]):
        value = float(scores[np.flatnonzero(calibration & (question_ids == question_id))[0]])
        values.append(value * (1.0 - value))
    values = np.asarray(values, dtype=np.float64)
    return [higher_quantile(values, q) for q in (0.25, 0.50, 0.75)]


def assigned_budget(risk: float, cutpoints: list[float]) -> int:
    uncertainty = risk * (1.0 - risk)
    if uncertainty < cutpoints[0]:
        return 1
    if uncertainty < cutpoints[1]:
        return 2
    if uncertainty < cutpoints[2]:
        return 4
    return 8


def prefix_event_token(tokenizer, row: dict) -> int:
    aliases = [
        normalize_answer(value)
        for value in (row.get("gold_answers") or [row["gold_answer"]])
    ]
    aliases = [value for value in aliases if value]
    tokens = list(map(int, row["generated_token_ids"]))
    for count in range(1, len(tokens) + 1):
        prefix = normalize_answer(
            tokenizer.decode(tokens[:count], skip_special_tokens=True)
        )
        if prefix and not any(alias.startswith(prefix) for alias in aliases):
            return count
    if not bool(row["model_correct"]):
        return max(len(tokens), 1)
    return -1


def candidate_risk(
    row_index: int,
    method: str | None,
    scores: dict[str, dict[int, np.ndarray]],
    states: dict[str, np.ndarray],
) -> float:
    if method is None:
        valid_stages = [
            stage for stage in STAGES if states[f"valid_T{stage}"][row_index]
        ]
        if not valid_stages:
            # Empty completions have no selected-token history. Use the
            # pre-generation maximum probability only as a deterministic
            # consensus tie-break; it is not an online branch decision.
            return -float(states["confidence_T0"][row_index, 0])
        stage = max(valid_stages)
        history_mean_logprob = float(states[f"confidence_T{stage}"][row_index, 3])
        return -history_mean_logprob
    valid = [
        float(scores[method][stage][row_index])
        for stage in STAGES
        if states[f"valid_T{stage}"][row_index]
    ]
    return max(valid) if valid else 1.0


def select_consensus(
    candidate_indices: list[int],
    rows: list[dict],
    method: str | None,
    scores: dict[str, dict[int, np.ndarray]],
    states: dict[str, np.ndarray],
) -> tuple[int, int]:
    clusters = defaultdict(list)
    for index in candidate_indices:
        clusters[normalize_answer(rows[index]["model_answer"])].append(index)
    ranked = []
    for answer, indices in clusters.items():
        risk = float(np.mean([
            candidate_risk(index, method, scores, states) for index in indices
        ]))
        ranked.append((-len(indices), risk, min(rows[index]["sample_index"] for index in indices), answer, indices))
    _, _, _, _, selected_cluster = min(ranked)
    selected = min(
        selected_cluster,
        key=lambda index: (
            candidate_risk(index, method, scores, states),
            rows[index]["sample_index"],
        ),
    )
    return selected, len(clusters)


def first_alert(
    row_index: int,
    method: str,
    scores: dict[str, dict[int, np.ndarray]],
    thresholds: dict[str, dict[int, float]],
    states: dict[str, np.ndarray],
) -> tuple[int | None, float]:
    margins = []
    length = int(states["answer_lengths"][row_index])
    for stage in STAGES:
        if not states[f"valid_T{stage}"][row_index] or stage >= length:
            continue
        margin = float(scores[method][stage][row_index] - thresholds[method][stage])
        margins.append(margin)
        if margin >= 0.0:
            return stage, max(margins)
    return None, max(margins) if margins else -float("inf")


def simulate_question(
    indices: list[int],
    budget: int,
    rows: list[dict],
    states: dict[str, np.ndarray],
    scores: dict[str, dict[int, np.ndarray]],
    thresholds: dict[str, dict[int, float]] | None,
    method: str | None,
) -> dict:
    selected_rows = indices[:budget]
    if method is None:
        retained = selected_rows
        tokens = int(sum(states["answer_lengths"][index] for index in retained))
        stopped = 0
    else:
        retained, rejected = [], []
        tokens = 0
        for index in selected_rows:
            alert, margin = first_alert(index, method, scores, thresholds, states)
            if alert is None:
                tokens += int(states["answer_lengths"][index])
                retained.append(index)
            else:
                tokens += alert
                rejected.append((margin, index, alert))
        stopped = len(rejected)
        if not retained:
            _, resumed, alert = min(
                rejected,
                key=lambda item: (item[0], rows[item[1]]["sample_index"]),
            )
            tokens += max(int(states["answer_lengths"][resumed]) - alert, 0)
            retained = [resumed]
    chosen, n_clusters = select_consensus(retained, rows, method, scores, states)
    return {
        "correct": int(bool(rows[chosen]["model_correct"])),
        "tokens": tokens,
        "retained": len(retained),
        "stopped": stopped,
        "clusters": n_clusters,
        "oracle_any_correct": int(any(bool(rows[index]["model_correct"]) for index in selected_rows)),
        "chosen_sample": int(rows[chosen]["sample_index"]),
    }


def bootstrap_mean(values: np.ndarray, seed: int, repeats: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(sampled, (0.025, 0.975))],
        "n_questions": int(len(values)),
    }


def summarize_policy(records: list[dict], seed: int, repeats: int) -> dict:
    return {
        "n_questions": len(records),
        "accuracy": bootstrap_mean(np.asarray([r["correct"] for r in records]), seed, repeats),
        "tokens": bootstrap_mean(np.asarray([r["tokens"] for r in records]), seed + 1, repeats),
        "retained_candidates_mean": float(np.mean([r["retained"] for r in records])),
        "stopped_candidates_mean": float(np.mean([r["stopped"] for r in records])),
        "attainable_any_correct": float(np.mean([r["oracle_any_correct"] for r in records])),
    }


def paired_contrast(left: list[dict], right: list[dict], seed: int, repeats: int) -> dict:
    return {
        "accuracy_delta": bootstrap_mean(
            np.asarray([a["correct"] - b["correct"] for a, b in zip(left, right)]),
            seed,
            repeats,
        ),
        "token_delta": bootstrap_mean(
            np.asarray([a["tokens"] - b["tokens"] for a, b in zip(left, right)]),
            seed + 1,
            repeats,
        ),
    }


def run_condition(args, condition: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(condition["tokenizer"], local_files_only=True)
    rows = read_jsonl(ROOT / condition["data"])
    states, _ = load_condition(tokenizer, ROOT / condition["data"], ROOT / condition["cache"])
    states = canonicalize(states)
    labels = states["labels"].astype(np.int8)
    question_ids = states["question_ids"].astype(str)
    pair_ids = states["pair_ids"].astype(str)
    fold_ids = group_fold_ids(pair_ids, seed=args.seed, folds=N_FOLDS)
    methods = ["confidence", "observed", "dense"]
    if "sae_T0" in states:
        methods.append("sae")
    all_methods = ["random", *methods]
    scores = {
        method: {stage: np.full(len(rows), np.nan) for stage in STAGES}
        for method in all_methods
    }
    thresholds_by_fold = {}
    t0_scores = np.full(len(rows), np.nan)
    t0_conf_scores = np.full(len(rows), np.nan)
    budget_cutpoints_by_fold = {}
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    for fold in range(N_FOLDS):
        calibration_fold = (fold + 1) % N_FOLDS
        fit = (fold_ids != fold) & (fold_ids != calibration_fold)
        calibration = fold_ids == calibration_fold
        test = fold_ids == fold

        t0_model = fit_t0(states, labels, fit, device, confidence_only=False)
        t0_conf_model = fit_t0(states, labels, fit, device, confidence_only=True)
        predicted_t0 = predict_t0(t0_model, states)
        predicted_t0_conf = predict_t0(t0_conf_model, states)
        t0_scores[test] = predicted_t0[test]
        t0_conf_scores[test] = predicted_t0_conf[test]
        budget_cutpoints_by_fold[fold] = {
            "fused": budget_cutpoints(predicted_t0, question_ids, calibration),
            "confidence": budget_cutpoints(predicted_t0_conf, question_ids, calibration),
        }

        fold_thresholds = {}
        calibration_scores = {method: {} for method in all_methods}
        for target_fpr in TARGET_FPRS:
            fold_thresholds[target_fpr] = {method: {} for method in all_methods}
        for stage in STAGES:
            for method in methods:
                model = fit_stage_readout(states, labels, fit, stage, method, device)
                predicted = predict_stage_readout(model, states, stage, method)
                scores[method][stage][test] = predicted[test]
                calibration_scores[method][stage] = predicted
            random_values = np.asarray([
                stable_uniform(
                    args.seed, condition["name"], question_ids[row_index],
                    rows[row_index]["sample_index"], stage,
                )
                for row_index in range(len(rows))
            ], dtype=np.float64)
            scores["random"][stage][test] = random_values[test]
            calibration_scores["random"][stage] = random_values

        # Calibrate one trajectory-level threshold after taking the maximum
        # score over all monitorable stages. This controls the family-wise
        # false-abort rate of the sequential policy rather than applying the
        # nominal rate independently three times.
        for method in all_methods:
            calibration_maxima = []
            for row_index in np.flatnonzero(calibration & (labels == 0)):
                local = [
                    calibration_scores[method][stage][row_index]
                    for stage in STAGES
                    if states[f"valid_T{stage}"][row_index]
                    and stage < int(states["answer_lengths"][row_index])
                ]
                if local:
                    calibration_maxima.append(max(local))
            for target_fpr in TARGET_FPRS:
                threshold = higher_quantile(
                    np.asarray(calibration_maxima, dtype=np.float64),
                    1.0 - target_fpr,
                )
                for stage in STAGES:
                    fold_thresholds[target_fpr][method][stage] = threshold
        thresholds_by_fold[fold] = fold_thresholds

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["question_id"]].append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: int(rows[index]["sample_index"]))
    event_tokens = np.asarray([prefix_event_token(tokenizer, row) for row in rows])

    fixed_records = {budget: [] for budget in FIXED_BUDGETS}
    policy_records = {
        f"{method}_gate_k8_fpr{int(target_fpr * 100):02d}": []
        for method in all_methods for target_fpr in TARGET_FPRS
    }
    for method in all_methods:
        for target_fpr in TARGET_FPRS:
            policy_records[f"{method}_gate_adaptive_fpr{int(target_fpr * 100):02d}"] = []
    policy_records["t0_adaptive_no_gate"] = []
    policy_records["t0_confidence_adaptive_no_gate"] = []
    policy_records["random_budget_no_gate"] = []

    per_policy_question_ids = []
    early = {
        f"{method}_fpr{int(target_fpr * 100):02d}": []
        for method in all_methods for target_fpr in TARGET_FPRS
    }
    alarms = {
        f"{method}_fpr{int(target_fpr * 100):02d}": []
        for method in all_methods for target_fpr in TARGET_FPRS
    }
    for question_id in sorted(grouped):
        indices = grouped[question_id]
        fold = int(fold_ids[indices[0]])
        per_policy_question_ids.append(question_id)
        fused_budget = assigned_budget(
            float(t0_scores[indices[0]]), budget_cutpoints_by_fold[fold]["fused"]
        )
        conf_budget = assigned_budget(
            float(t0_conf_scores[indices[0]]), budget_cutpoints_by_fold[fold]["confidence"]
        )
        random_budget = FIXED_BUDGETS[min(int(stable_uniform(args.seed, condition["name"], question_id, "budget") * 4), 3)]
        for budget in FIXED_BUDGETS:
            fixed_records[budget].append(
                simulate_question(indices, budget, rows, states, scores, None, None)
            )
        policy_records["t0_adaptive_no_gate"].append(
            simulate_question(indices, fused_budget, rows, states, scores, None, None)
        )
        policy_records["t0_confidence_adaptive_no_gate"].append(
            simulate_question(indices, conf_budget, rows, states, scores, None, None)
        )
        policy_records["random_budget_no_gate"].append(
            simulate_question(indices, random_budget, rows, states, scores, None, None)
        )
        for method in all_methods:
            for target_fpr in TARGET_FPRS:
                label = f"{method}_fpr{int(target_fpr * 100):02d}"
                thresholds = thresholds_by_fold[fold][target_fpr]
                policy_records[f"{method}_gate_k8_fpr{int(target_fpr * 100):02d}"].append(
                    simulate_question(indices, 8, rows, states, scores, thresholds, method)
                )
                policy_records[f"{method}_gate_adaptive_fpr{int(target_fpr * 100):02d}"].append(
                    simulate_question(indices, fused_budget, rows, states, scores, thresholds, method)
                )
                for index in indices:
                    alert, _ = first_alert(index, method, scores, thresholds, states)
                    alarms[label].append({
                        "question_id": question_id,
                        "error": int(labels[index] == 1),
                        "alert": int(alert is not None),
                    })
                    if labels[index] != 1:
                        continue
                    event_token = int(event_tokens[index])
                    early[label].append({
                        "question_id": question_id,
                        "strict_pre_event": int(alert is not None and event_token > 0 and alert < event_token),
                        "eligible": int(event_token > 1),
                        "strict_pre_event_eligible": int(alert is not None and event_token > 1 and alert < event_token),
                        "lead": (event_token - alert) if alert is not None and event_token > 0 and alert < event_token else None,
                    })

    fixed_summary = {
        f"fixed_k{budget}": summarize_policy(records, args.seed + budget, args.bootstrap)
        for budget, records in fixed_records.items()
    }
    policy_summary = {
        name: summarize_policy(records, args.seed + 100 + index, args.bootstrap)
        for index, (name, records) in enumerate(sorted(policy_records.items()))
    }
    contrasts = {}
    fixed_k4 = fixed_records[4]
    for index, (name, records) in enumerate(sorted(policy_records.items())):
        contrasts[f"{name}_minus_fixed_k4"] = paired_contrast(
            records, fixed_k4, args.seed + 1000 + index * 2, args.bootstrap
        )
    for method in methods:
        name = f"{method}_gate_adaptive_fpr10"
        random_name = "random_gate_adaptive_fpr10"
        contrasts[f"{name}_minus_{random_name}"] = paired_contrast(
            policy_records[name], policy_records[random_name],
            args.seed + 3000 + len(contrasts), args.bootstrap,
        )
        if method != "confidence":
            confidence_name = "confidence_gate_adaptive_fpr10"
            contrasts[f"{name}_minus_{confidence_name}"] = paired_contrast(
                policy_records[name], policy_records[confidence_name],
                args.seed + 4000 + len(contrasts), args.bootstrap,
            )

    early_summary = {}
    for name, records in early.items():
        strict = np.asarray([row["strict_pre_event"] for row in records], dtype=np.float64)
        eligible = [row for row in records if row["eligible"]]
        strict_eligible = np.asarray(
            [row["strict_pre_event_eligible"] for row in eligible], dtype=np.float64
        )
        leads = [row["lead"] for row in records if row["lead"] is not None]
        early_summary[name] = {
            "n_error_trajectories": len(records),
            "strict_pre_event_recall": float(strict.mean()) if len(strict) else None,
            "n_eligible_after_first_token": len(eligible),
            "eligible_pre_event_recall": float(strict_eligible.mean()) if len(strict_eligible) else None,
            "mean_token_lead": float(np.mean(leads)) if leads else None,
        }

    operating_points = {}
    for name, records in alarms.items():
        correct = [row for row in records if not row["error"]]
        errors = [row for row in records if row["error"]]
        operating_points[name] = {
            "n_correct_trajectories": len(correct),
            "realized_false_abort_rate": float(np.mean([row["alert"] for row in correct])) if correct else None,
            "n_error_trajectories": len(errors),
            "error_alert_rate": float(np.mean([row["alert"] for row in errors])) if errors else None,
        }

    return {
        "condition": condition,
        "n_questions": len(grouped),
        "n_trajectories": len(rows),
        "base_accuracy": float(np.mean(labels == 0)),
        "methods": all_methods,
        "fixed_budget": fixed_summary,
        "policies": policy_summary,
        "contrasts": contrasts,
        "operating_points": operating_points,
        "early_detection": early_summary,
        "budget_distribution": {
            name: dict(Counter(record["retained"] + record["stopped"] for record in records))
            for name, records in policy_records.items()
            if name.endswith("no_gate")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--conditions", nargs="*", default=["gemma", "qwen15"],
        choices=["gemma", "qwen15"],
    )
    args = parser.parse_args()
    specifications = {
        "gemma": {
            "name": "Gemma-2-2B TriviaQA",
            "tokenizer": "google/gemma-2-2b",
            "data": "data/fpe14_gemma_trivia_confirmatory_trajectory.jsonl",
            "cache": "outputs/cache/fpe14_gemma_trivia_confirmatory_trajectory_states.npz",
        },
        "qwen15": {
            "name": "Qwen2.5-1.5B-Instruct TriviaQA",
            "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
            "data": "data/fpe14_qwen15_trivia_confirmatory_trajectory.jsonl",
            "cache": "outputs/cache/fpe14_qwen15_trivia_confirmatory_trajectory_states.npz",
        },
    }
    payload = {
        "experiment": "NN36 stage-wise budget-constrained intervention",
        "status": "retrospective cache-only constructive validation",
        "seed": args.seed,
        "target_correct_trajectory_false_abort_rates": TARGET_FPRS,
        "candidate_budgets": FIXED_BUDGETS,
        "conditions": [],
    }
    for name in args.conditions:
        print(f"[NN36] running {name}", flush=True)
        payload["conditions"].append(run_condition(args, specifications[name]))
    output = ROOT / "outputs" / "poc_nn36_stage_budget_intervention_results.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output}")
    for condition in payload["conditions"]:
        print("\n", condition["condition"]["name"])
        for key in ("fixed_k1", "fixed_k2", "fixed_k4", "fixed_k8"):
            row = condition["fixed_budget"][key]
            print(key, row["accuracy"]["mean"], row["tokens"]["mean"])
        for key in (
            "random_gate_adaptive_fpr10", "confidence_gate_adaptive_fpr10",
            "observed_gate_adaptive_fpr10", "dense_gate_adaptive_fpr10",
            "sae_gate_adaptive_fpr10",
        ):
            if key in condition["policies"]:
                row = condition["policies"][key]
                print(key, row["accuracy"]["mean"], row["tokens"]["mean"])


if __name__ == "__main__":
    main()
