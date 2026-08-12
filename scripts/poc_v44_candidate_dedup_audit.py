"""Audit V44 candidate deduplication for option bias and label leakage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import (  # noqa: E402
    accuracy_from_indices,
    bootstrap_delta,
    center_by_question,
    load_npz,
)
from scripts.poc_v44_framework_guided_cevr_finetuning import (  # noqa: E402
    UniqueCandidates,
    collapse_candidates,
    first_choice,
    mixed_question_mask,
    selection_accuracy,
    unique_choice,
    within_question_auroc,
)


def logistic_oof(
    data: UniqueCandidates, correct: np.ndarray
) -> tuple[np.ndarray, dict]:
    mask = mixed_question_mask(data)
    x = center_by_question(data.x[mask], data.question_ids[mask])
    y = correct[mask]
    qids = data.question_ids[mask]
    predictions = np.zeros(len(y), dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for train_index, valid_index in splitter.split(x, y, qids):
        scaler = StandardScaler()
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=4000,
            random_state=20260808,
        )
        classifier.fit(scaler.fit_transform(x[train_index]), y[train_index])
        predictions[valid_index] = classifier.predict_proba(
            scaler.transform(x[valid_index])
        )[:, 1]
    return predictions, {
        "question_accuracy": selection_accuracy(y, qids, predictions),
        "candidate_auroc": float(roc_auc_score(y, predictions)),
        "within_question_auroc": within_question_auroc(y, qids, predictions),
        "n_rows": int(len(y)),
        "n_questions": int(len(np.unique(qids))),
    }


def permute_within_question(
    data: UniqueCandidates, rng: np.random.Generator
) -> np.ndarray:
    output = data.correct.copy()
    for qid in np.unique(data.question_ids):
        indices = np.flatnonzero(data.question_ids == qid)
        if len(np.unique(output[indices])) != 2:
            continue
        output[indices] = 0
        output[int(rng.choice(indices))] = 1
    return output


def option_prior_choice(
    train: UniqueCandidates, target: UniqueCandidates
) -> tuple[dict[str, int], dict[str, float]]:
    rates = {}
    global_rate = float(train.correct.mean())
    for option in sorted(set(train.options)):
        values = train.correct[train.options == option]
        rates[str(option)] = float((values.sum() + global_rate) / (len(values) + 1))
    utility = np.asarray([rates.get(str(option), global_rate) for option in target.options])
    return unique_choice(target, utility), rates


def candidate_count_summary(data: UniqueCandidates) -> dict:
    counts = Counter(Counter(data.question_ids).values())
    return {
        "n_unique_candidates": int(len(data.correct)),
        "mean_unique_candidates_per_question": float(
            len(data.correct) / len(np.unique(data.question_ids))
        ),
        "count_distribution": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--output", default="outputs/poc_v44_candidate_dedup_audit_results.json")
    args = parser.parse_args()

    train_cache = load_npz(ROOT / "outputs/cache/v40f_no_rationale_states.npz")
    train = collapse_candidates(
        train_cache,
        ROOT / "outputs/poc_v40b_commitment_generation_results.json",
        "raw_L18",
        clean=True,
    )
    _, real = logistic_oof(train, train.correct)
    rng = np.random.default_rng(20260808)
    null = []
    for permutation in range(args.permutations):
        permuted = permute_within_question(train, rng)
        _, metrics = logistic_oof(train, permuted)
        null.append(metrics)
        print(f"permutation={permutation + 1}/{args.permutations} {metrics}", flush=True)

    audit_targets = []
    for name, cache_path, generation_path in (
        ("V41", "outputs/cache/v41_hotpot_no_rationale_states.npz",
         "outputs/poc_v41_hotpot_no_rationale_generation_results.json"),
        ("V42", "outputs/cache/v42_hotpot_no_rationale_states.npz",
         "outputs/poc_v42_hotpot_no_rationale_generation_results.json"),
    ):
        cache = load_npz(ROOT / cache_path)
        target = collapse_candidates(cache, ROOT / generation_path, "raw_C_L18", clean=False)
        choice, rates = option_prior_choice(train, target)
        first = first_choice(cache)
        labels = cache["labels"].astype(int)
        result = accuracy_from_indices(labels, choice)
        result["paired_vs_first"] = bootstrap_delta(labels, first, choice, 20260900)
        audit_targets.append({
            "dataset": name,
            "candidate_counts": candidate_count_summary(target),
            "option_prior_rates_from_v40": rates,
            "option_prior_result": result,
        })

    null_accuracy = np.asarray([row["question_accuracy"] for row in null])
    null_auroc = np.asarray([row["within_question_auroc"] for row in null])
    payload = {
        "experiment": "poc_v44_candidate_dedup_audit",
        "purpose": "exclude option-letter bias and obvious label leakage as explanations for V44",
        "train_candidate_counts": candidate_count_summary(train),
        "real_v40_oof_unique_centered_logistic": real,
        "within_question_label_permutation": {
            "n_permutations": args.permutations,
            "question_accuracy_mean": float(null_accuracy.mean()),
            "question_accuracy_range": [float(null_accuracy.min()), float(null_accuracy.max())],
            "within_question_auroc_mean": float(null_auroc.mean()),
            "within_question_auroc_range": [float(null_auroc.min()), float(null_auroc.max())],
            "empirical_p_question_accuracy": float(
                (1 + np.sum(null_accuracy >= real["question_accuracy"]))
                / (args.permutations + 1)
            ),
            "empirical_p_within_question_auroc": float(
                (1 + np.sum(null_auroc >= real["within_question_auroc"]))
                / (args.permutations + 1)
            ),
            "rows": null,
        },
        "targets": audit_targets,
        "interpretation_boundary": [
            "A null permutation result rules out direct label leakage in this grouped pipeline; it is not a causal proof.",
            "The option-only baseline tests answer-position imbalance but not answer-text priors.",
            "Candidate deduplication changes the router's comparison set weighting; it does not add an unsampled answer.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "real": real,
        "null": payload["within_question_label_permutation"],
        "targets": audit_targets,
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
