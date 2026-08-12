"""V40e: paired question bootstrap for full-context versus answer-only states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v37_same_question_trajectory import (
    SEEDS,
    center_within_question,
    fit_predict,
)


METHODS = (
    ("raw_residual", 9, "raw"),
    ("raw_residual", 18, "raw"),
    ("sae", 9, "sae"),
    ("sae", 18, "sae"),
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def oof_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray,
    kind: str,
    seed: int,
) -> np.ndarray:
    valid = np.ones(len(labels), dtype=bool)
    centered = center_within_question(features, question_ids, valid)
    unique_questions = np.unique(question_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_questions.copy()
    rng.shuffle(shuffled)
    fold_map = {question_id: index % 5 for index, question_id in enumerate(shuffled)}
    predictions = np.full(len(labels), np.nan)
    for fold in range(5):
        test = np.asarray([fold_map[question_id] == fold for question_id in question_ids])
        train = ~test
        predictions[test] = fit_predict(
            centered[train], labels[train], centered[test], kind, seed + fold
        )
    return predictions


def question_aurocs(
    labels: np.ndarray,
    predictions: np.ndarray,
    question_ids: np.ndarray,
) -> dict[str, float]:
    output = {}
    for question_id in np.unique(question_ids):
        mask = question_ids == question_id
        if len(np.unique(labels[mask])) == 2:
            if np.ptp(predictions[mask]) <= 1e-10:
                output[question_id] = 0.5
            else:
                output[question_id] = float(roc_auc_score(labels[mask], predictions[mask]))
    return output


def bootstrap(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> dict:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "bootstrap_probability_le_zero": float(np.mean(means <= 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-cache", default="outputs/cache/v40b_distractor_evidence_states_clean.npz"
    )
    parser.add_argument(
        "--answer-cache", default="outputs/cache/v40d_answer_only_states.npz"
    )
    parser.add_argument(
        "--no-rationale-cache", default="outputs/cache/v40f_no_rationale_states.npz"
    )
    parser.add_argument(
        "--output", default="outputs/poc_v40e_context_increment_bootstrap_results.json"
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    full = load_npz(ROOT / args.full_cache)
    answer = load_npz(ROOT / args.answer_cache)
    no_rationale = load_npz(ROOT / args.no_rationale_cache)
    labels = full["labels"].astype(int)
    question_ids = full["question_ids"].astype(str)
    if not np.array_equal(labels, answer["labels"].astype(int)):
        raise ValueError("Label order differs between full and answer-only caches")
    if not np.array_equal(question_ids, answer["question_ids"].astype(str)):
        raise ValueError("Question order differs between full and answer-only caches")
    if not np.array_equal(labels, no_rationale["labels"].astype(int)):
        raise ValueError("Label order differs between full and no-rationale caches")
    if not np.array_equal(question_ids, no_rationale["question_ids"].astype(str)):
        raise ValueError("Question order differs between full and no-rationale caches")

    results = []
    value_store = {}
    for method, layer, kind in METHODS:
        full_features = full[f"{'raw' if method == 'raw_residual' else 'sae'}_C_L{layer}"].astype(np.float32)
        answer_features = answer[f"{'raw' if method == 'raw_residual' else 'sae'}_L{layer}"].astype(np.float32)
        no_rationale_features = no_rationale[
            f"{'raw' if method == 'raw_residual' else 'sae'}_L{layer}"
        ].astype(np.float32)
        full_by_seed = []
        answer_by_seed = []
        no_rationale_by_seed = []
        common_questions = None
        for seed in SEEDS:
            full_predictions = oof_predictions(
                full_features, labels, question_ids, kind, seed
            )
            answer_predictions = oof_predictions(
                answer_features, labels, question_ids, kind, seed
            )
            no_rationale_predictions = oof_predictions(
                no_rationale_features, labels, question_ids, kind, seed
            )
            full_aurocs = question_aurocs(labels, full_predictions, question_ids)
            answer_aurocs = question_aurocs(labels, answer_predictions, question_ids)
            no_rationale_aurocs = question_aurocs(
                labels, no_rationale_predictions, question_ids
            )
            current = sorted(
                set(full_aurocs) & set(answer_aurocs) & set(no_rationale_aurocs)
            )
            common_questions = current if common_questions is None else sorted(
                set(common_questions) & set(current)
            )
            full_by_seed.append(full_aurocs)
            answer_by_seed.append(answer_aurocs)
            no_rationale_by_seed.append(no_rationale_aurocs)

        full_values = np.asarray([
            np.mean([seed_values[qid] for seed_values in full_by_seed])
            for qid in common_questions
        ])
        answer_values = np.asarray([
            np.mean([seed_values[qid] for seed_values in answer_by_seed])
            for qid in common_questions
        ])
        no_rationale_values = np.asarray([
            np.mean([seed_values[qid] for seed_values in no_rationale_by_seed])
            for qid in common_questions
        ])
        delta = full_values - answer_values
        rationale_delta = full_values - no_rationale_values
        value_store[(method, layer)] = {
            "full": full_values,
            "answer_only": answer_values,
            "no_rationale": no_rationale_values,
        }
        rng = np.random.default_rng(20260714 + layer + (100 if method == "sae" else 0))
        results.append({
            "method": method,
            "layer": layer,
            "n_discordant_questions": len(common_questions),
            "full_context": bootstrap(full_values, rng, args.n_bootstrap),
            "answer_only": bootstrap(answer_values, rng, args.n_bootstrap),
            "no_rationale": bootstrap(no_rationale_values, rng, args.n_bootstrap),
            "paired_delta": bootstrap(delta, rng, args.n_bootstrap),
            "rationale_increment": bootstrap(rationale_delta, rng, args.n_bootstrap),
        })

    representation_comparisons = []
    for layer in (9, 18):
        raw_values = value_store[("raw_residual", layer)]
        sae_values = value_store[("sae", layer)]
        for condition in ("full", "answer_only", "no_rationale"):
            delta = raw_values[condition] - sae_values[condition]
            rng = np.random.default_rng(20260714 + layer + len(condition))
            representation_comparisons.append({
                "layer": layer,
                "condition": condition,
                "contrast": "raw_residual minus SAE",
                "paired_delta": bootstrap(delta, rng, args.n_bootstrap),
            })

    payload = {
        "experiment": "poc_v40e_context_increment_bootstrap",
        "comparison": "full V40b C-stage versus answer-only commitment",
        "unit": "discordant question",
        "folding": "same 5-fold question-grouped splits, averaged over three seeds",
        "n_bootstrap": args.n_bootstrap,
        "results": results,
        "representation_comparisons": representation_comparisons,
    }
    output_path = ROOT / args.output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output_path}")
    for row in results:
        print(
            f"{row['method']:<12} L{row['layer']} "
            f"full={row['full_context']['mean']:.4f} "
            f"answer={row['answer_only']['mean']:.4f} "
            f"no_rat={row['no_rationale']['mean']:.4f} "
            f"delta={row['paired_delta']['mean']:.4f} "
            f"CI={row['paired_delta']['ci95']} "
            f"rat_delta={row['rationale_increment']['mean']:.4f} "
            f"rat_CI={row['rationale_increment']['ci95']}"
        )
    for row in representation_comparisons:
        print(
            f"raw-sae L{row['layer']} {row['condition']:<12} "
            f"delta={row['paired_delta']['mean']:.4f} "
            f"CI={row['paired_delta']['ci95']}"
        )


if __name__ == "__main__":
    main()
