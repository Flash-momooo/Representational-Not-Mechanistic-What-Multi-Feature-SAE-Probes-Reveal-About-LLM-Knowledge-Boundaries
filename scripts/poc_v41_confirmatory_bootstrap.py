"""V41 fixed-protocol question bootstrap and confirmatory decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v37_same_question_trajectory import SEEDS
from scripts.poc_v40e_context_increment_bootstrap import (
    load_npz,
    oof_predictions,
    question_aurocs,
)


METHODS = (
    ("raw_residual", 9, "raw"),
    ("raw_residual", 18, "raw"),
    ("sae", 9, "sae"),
    ("sae", 18, "sae"),
)


def bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    reference: float,
) -> dict:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "reference": reference,
        "bootstrap_probability_at_or_below_reference": float(np.mean(means <= reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-cache", default="outputs/cache/v41_hotpot_no_rationale_states.npz"
    )
    parser.add_argument(
        "--answer-cache", default="outputs/cache/v41_hotpot_answer_only_states.npz"
    )
    parser.add_argument(
        "--output", default="outputs/poc_v41_hotpot_confirmatory_bootstrap_results.json"
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    full = load_npz(ROOT / args.full_cache)
    answer = load_npz(ROOT / args.answer_cache)
    labels = full["labels"].astype(int)
    question_ids = full["question_ids"].astype(str)
    if not np.array_equal(labels, answer["labels"].astype(int)):
        raise ValueError("Label order differs between caches")
    if not np.array_equal(question_ids, answer["question_ids"].astype(str)):
        raise ValueError("Question order differs between caches")

    results = []
    value_store = {}
    for method, layer, kind in METHODS:
        prefix = "raw" if method == "raw_residual" else "sae"
        full_features = full[f"{prefix}_C_L{layer}"].astype(np.float32)
        answer_features = answer[f"{prefix}_L{layer}"].astype(np.float32)
        full_by_seed = []
        answer_by_seed = []
        common_questions = None
        for seed in SEEDS:
            full_predictions = oof_predictions(
                full_features, labels, question_ids, kind, seed
            )
            answer_predictions = oof_predictions(
                answer_features, labels, question_ids, kind, seed
            )
            full_aurocs = question_aurocs(labels, full_predictions, question_ids)
            answer_aurocs = question_aurocs(labels, answer_predictions, question_ids)
            current = sorted(set(full_aurocs) & set(answer_aurocs))
            common_questions = current if common_questions is None else sorted(
                set(common_questions) & set(current)
            )
            full_by_seed.append(full_aurocs)
            answer_by_seed.append(answer_aurocs)

        full_values = np.asarray([
            np.mean([seed_values[question_id] for seed_values in full_by_seed])
            for question_id in common_questions
        ])
        answer_values = np.asarray([
            np.mean([seed_values[question_id] for seed_values in answer_by_seed])
            for question_id in common_questions
        ])
        delta = full_values - answer_values
        value_store[(method, layer)] = {
            "full": full_values,
            "answer_only": answer_values,
        }
        rng = np.random.default_rng(20260714 + layer + (100 if method == "sae" else 0))
        results.append({
            "method": method,
            "layer": layer,
            "n_discordant_questions": len(common_questions),
            "full_context": bootstrap(full_values, rng, args.n_bootstrap, 0.5),
            "answer_only": bootstrap(answer_values, rng, args.n_bootstrap, 0.5),
            "context_increment": bootstrap(delta, rng, args.n_bootstrap, 0.0),
        })

    representation_comparisons = []
    for layer in (9, 18):
        delta = (
            value_store[("raw_residual", layer)]["full"]
            - value_store[("sae", layer)]["full"]
        )
        rng = np.random.default_rng(20260814 + layer)
        representation_comparisons.append({
            "layer": layer,
            "contrast": "full-context raw residual minus SAE",
            "paired_delta": bootstrap(delta, rng, args.n_bootstrap, 0.0),
        })

    by_key = {(row["method"], row["layer"]): row for row in results}
    raw_l18 = by_key[("raw_residual", 18)]
    sae_l18 = by_key[("sae", 18)]
    raw_strong = (
        raw_l18["full_context"]["mean"] >= 0.65
        and raw_l18["full_context"]["ci95"][0] > 0.5
    )
    sae_strong = (
        sae_l18["full_context"]["mean"] >= 0.60
        and sae_l18["full_context"]["ci95"][0] > 0.5
    )
    context_replication = (
        raw_l18["context_increment"]["ci95"][0] > 0.0
        and sae_l18["context_increment"]["ci95"][0] > 0.0
    )
    decision = {
        "raw_l18_strong_replication": raw_strong,
        "sae_l18_strong_replication": sae_strong,
        "context_dependence_replication": context_replication,
        "overall_post_commitment_decision": (
            "strong_replication" if raw_strong and sae_strong and context_replication
            else "weak_or_failed_replication"
        ),
        "pre_commitment_decision": "not_tested_in_v41_after_pre-extraction_amendment",
    }
    payload = {
        "experiment": "poc_v41_hotpot_confirmatory_bootstrap",
        "protocol": "frozen L9/L18, question-grouped 5-fold CV, three seeds",
        "n_rows": len(labels),
        "n_questions": len(np.unique(question_ids)),
        "n_bootstrap": args.n_bootstrap,
        "results": results,
        "representation_comparisons": representation_comparisons,
        "confirmatory_decision": decision,
    }
    output_path = ROOT / args.output
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output_path}")
    for row in results:
        print(
            f"{row['method']:<12} L{row['layer']} "
            f"full={row['full_context']['mean']:.4f} "
            f"CI={row['full_context']['ci95']} "
            f"answer={row['answer_only']['mean']:.4f} "
            f"delta={row['context_increment']['mean']:.4f} "
            f"delta_CI={row['context_increment']['ci95']}"
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
