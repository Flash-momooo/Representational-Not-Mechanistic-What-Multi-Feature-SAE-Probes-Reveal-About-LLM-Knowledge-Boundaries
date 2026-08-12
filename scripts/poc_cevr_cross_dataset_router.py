"""CEVR: cross-dataset post-commitment candidate verification router.

Train a fixed, question-centered error readout on V40f (2Wiki) and apply it
without tuning to V41 (HotpotQA).  Candidate states are read only after an
answer candidate has been appended to evidence and the question.  This is a
candidate verifier, not a predictor of a future ungenerated branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PRIMARY = ("raw", "raw_residual")
AUXILIARY = ("sae", "sae")
N_BOOTSTRAP = 10_000


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def center_by_question(features: np.ndarray, question_ids: np.ndarray) -> np.ndarray:
    """Keep only candidate-relative state within the same evidence question."""
    output = features.astype(np.float32, copy=True)
    for question_id in np.unique(question_ids):
        mask = question_ids == question_id
        output[mask] -= output[mask].mean(axis=0, keepdims=True)
    return output


def fit_router(
    train: dict[str, np.ndarray],
    feature_key: str,
) -> tuple[StandardScaler, LogisticRegression]:
    x = center_by_question(train[feature_key], train["question_ids"].astype(str))
    y = train["labels"].astype(int)
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        class_weight="balanced",
        max_iter=4000,
        random_state=20260801,
    )
    classifier.fit(scaler.fit_transform(x), y)
    return scaler, classifier


def risk_scores(
    train: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    train_feature_key: str,
    target_feature_key: str,
) -> np.ndarray:
    scaler, classifier = fit_router(train, train_feature_key)
    x = center_by_question(target[target_feature_key], target["question_ids"].astype(str))
    return classifier.predict_proba(scaler.transform(x))[:, 1]


def read_generation_scores(path: Path, question_ids: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    values = {
        (str(row["item_id"]), int(row["sample_index"])): float(
            row["candidate_mean_logprob"][row["selected_option"]]
        )
        for row in rows
    }
    output = []
    for question_id, sample_index in zip(question_ids.astype(str), sample_indices.astype(int)):
        key = (question_id, int(sample_index))
        if key not in values:
            raise ValueError(f"Missing generator likelihood for {key}")
        output.append(values[key])
    return np.asarray(output, dtype=np.float64)


def choose_per_question(
    question_ids: np.ndarray,
    sample_indices: np.ndarray,
    utility: np.ndarray,
) -> dict[str, int]:
    """Choose maximum utility; deterministic earliest-sample tie break."""
    selected: dict[str, int] = {}
    for question_id in np.unique(question_ids):
        indices = np.flatnonzero(question_ids == question_id)
        order = np.lexsort((sample_indices[indices], -utility[indices]))
        selected[str(question_id)] = int(indices[order[0]])
    return selected


def accuracy_from_indices(labels: np.ndarray, selected: dict[str, int]) -> dict[str, float | list[int]]:
    indices = np.asarray(list(selected.values()), dtype=int)
    success = 1 - labels[indices].astype(int)
    return {
        "accuracy": float(success.mean()),
        "n_questions": int(len(indices)),
        "correct_questions": int(success.sum()),
        "selected_indices": [int(index) for index in indices],
    }


def bootstrap_delta(
    labels: np.ndarray,
    first: dict[str, int],
    challenger: dict[str, int],
    seed: int,
) -> dict:
    question_ids = sorted(first)
    first_values = np.asarray([1 - int(labels[first[q]]) for q in question_ids])
    challenger_values = np.asarray([1 - int(labels[challenger[q]]) for q in question_ids])
    delta = challenger_values - first_values
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(question_ids), size=(N_BOOTSTRAP, len(question_ids)))
    means = delta[sample_indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
    }


def candidate_metrics(labels: np.ndarray, risk: np.ndarray, question_ids: np.ndarray) -> dict:
    per_question = []
    for question_id in np.unique(question_ids):
        indices = np.flatnonzero(question_ids == question_id)
        y = labels[indices]
        if len(np.unique(y)) == 2:
            per_question.append(float(roc_auc_score(y, risk[indices])))
    return {
        "population_auroc": float(roc_auc_score(labels, risk)),
        "population_auprc": float(average_precision_score(labels, risk)),
        "within_question_auroc": float(np.mean(per_question)),
        "n_discordant_questions": int(len(per_question)),
    }


def candidate_coverage(labels: np.ndarray, question_ids: np.ndarray) -> dict:
    values = []
    for question_id in np.unique(question_ids):
        indices = np.flatnonzero(question_ids == question_id)
        values.append(int(np.any(labels[indices] == 0)))
    return {
        "reachable_oracle_accuracy": float(np.mean(values)),
        "questions_with_a_correct_sample": int(sum(values)),
        "n_questions": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--target-cache", default="outputs/cache/v41_hotpot_no_rationale_states.npz")
    parser.add_argument("--target-generation", default="outputs/poc_v41_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--output", default="outputs/poc_cevr_cross_dataset_router_results.json")
    parser.add_argument("--protocol", default="paper/CEVR_CROSS_DATASET_ROUTER_PROTOCOL.md")
    parser.add_argument(
        "--status",
        default="retrospective_cross_dataset_analysis; V41 was previously inspected for detection, not prospective routing confirmation",
    )
    args = parser.parse_args()

    train = load_npz(ROOT / args.train_cache)
    target = load_npz(ROOT / args.target_cache)
    labels = target["labels"].astype(int)
    question_ids = target["question_ids"].astype(str)
    sample_indices = target["sample_indices"].astype(int)
    if set(sample_indices) != set(range(8)):
        raise ValueError("Frozen CEVR target requires exactly eight samples per question")
    counts = Counter(question_ids)
    if set(counts.values()) != {8}:
        raise ValueError("Frozen CEVR target requires eight rows per question")

    first = {
        str(question_id): int(np.flatnonzero(
            (question_ids == question_id) & (sample_indices == 0)
        )[0])
        for question_id in np.unique(question_ids)
    }
    likelihood = read_generation_scores(ROOT / args.target_generation, question_ids, sample_indices)
    likelihood_choice = choose_per_question(question_ids, sample_indices, likelihood)
    results = {
        "first_sample": accuracy_from_indices(labels, first),
        "restricted_likelihood": accuracy_from_indices(labels, likelihood_choice),
        "reachable_oracle": candidate_coverage(labels, question_ids),
    }
    methods = []
    for prefix, name in (PRIMARY, AUXILIARY):
        feature_key_train = f"{prefix}_L18"
        feature_key_target = f"{prefix}_C_L18"
        risk = risk_scores(train, target, feature_key_train, feature_key_target)
        selection = choose_per_question(question_ids, sample_indices, -risk)
        result = accuracy_from_indices(labels, selection)
        result["paired_vs_first"] = bootstrap_delta(
            labels, first, selection, 20260801 + (100 if prefix == "sae" else 0)
        )
        result["candidate_level"] = candidate_metrics(labels, risk, question_ids)
        results[name] = result
        methods.append({
            "name": name,
            "feature": "Gemma-2-2B layer-18 post-commitment " + prefix,
            "selection_accuracy": result["accuracy"],
            "delta_vs_first": result["paired_vs_first"],
            "candidate_level": result["candidate_level"],
        })

    likelihood_result = results["restricted_likelihood"]
    likelihood_result["paired_vs_first"] = bootstrap_delta(
        labels, first, likelihood_choice, 20260851
    )
    raw_delta = results["raw_residual"]["paired_vs_first"]
    raw_choice = {
        str(question_ids[index]): int(index)
        for index in results["raw_residual"]["selected_indices"]
    }
    payload = {
        "experiment": "poc_cevr_cross_dataset_router",
        "protocol": args.protocol,
        "status": args.status,
        "train": {
            "dataset": "V40f / 2WikiMultiHopQA",
            "n_rows": int(len(train["labels"])),
            "n_questions": int(len(np.unique(train["question_ids"].astype(str)))),
            "risk_rate": float(train["labels"].mean()),
        },
        "target": {
            "dataset": "V41 / HotpotQA distractor validation",
            "n_rows": int(len(labels)),
            "n_questions": int(len(np.unique(question_ids))),
            "risk_rate": float(labels.mean()),
            "samples_per_question": 8,
        },
        "results": results,
        "method_summary": methods,
        "secondary_comparisons": {
            "raw_l18_vs_restricted_likelihood": bootstrap_delta(
                labels, likelihood_choice, raw_choice, 20260871
            ),
        },
        "primary_decision": {
            "raw_l18_delta_vs_first": raw_delta,
            "passes_positive_paired_ci": bool(raw_delta["ci95"][0] > 0.0),
            "scope": "candidate routing after commitment under evidence-grounded constrained answers",
        },
        "limitations": [
            "The candidate set is constrained and contains a correct candidate only on reachable-oracle questions.",
            "The method verifies post-commitment compatibility; it does not predict which ungenerated stochastic branch will fail.",
            "V41 is an independent dataset but this routing analysis is retrospective because its detection results were already inspected.",
            "Dense raw residual is the primary representation; SAE is an explicitly secondary sparse comparison.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "first": results["first_sample"]["accuracy"],
        "likelihood": results["restricted_likelihood"]["accuracy"],
        "raw": results["raw_residual"]["accuracy"],
        "sae": results["sae"]["accuracy"],
        "oracle": results["reachable_oracle"]["reachable_oracle_accuracy"],
        "raw_delta": raw_delta,
        "output": str(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
