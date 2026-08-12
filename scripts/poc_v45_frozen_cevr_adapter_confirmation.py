"""V45 prospective confirmation of the V40-selected CEVR adapter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import (  # noqa: E402
    accuracy_from_indices,
    bootstrap_delta,
    center_by_question,
    choose_per_question,
    load_npz,
    risk_scores,
)
from scripts.poc_v44_framework_guided_cevr_finetuning import (  # noqa: E402
    collapse_candidates,
    first_choice,
    fit_ensemble,
    mixed_question_mask,
    unique_choice,
    within_question_auroc,
)


def coverage(labels: np.ndarray, question_ids: np.ndarray) -> dict:
    reachable = []
    for qid in np.unique(question_ids):
        reachable.append(bool(np.any(labels[question_ids == qid] == 0)))
    return {
        "accuracy": float(np.mean(reachable)),
        "questions": int(np.sum(reachable)),
        "n_questions": int(len(reachable)),
    }


def source_isolation_audit(target_items: Path) -> dict:
    rows = [json.loads(line) for line in target_items.read_text(encoding="utf-8").splitlines() if line]
    target_ids = {str(row["pair_id"]) for row in rows}
    target_ids.update(
        str(source) for row in rows for source in row["distractor_evidence_sources"]
    )
    pattern = re.compile(r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])")
    prior_ids: set[str] = set()
    for path in (ROOT / "data").glob("*.jsonl"):
        if path.resolve() != target_items.resolve():
            prior_ids.update(pattern.findall(path.read_text(encoding="utf-8")))
    return {
        "n_target_source_ids": int(len(target_ids)),
        "n_overlapping_prior_source_ids": int(len(target_ids & prior_ids)),
        "passes_zero_overlap": not bool(target_ids & prior_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--train-generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--target-cache", default="outputs/cache/v45_hotpot_no_rationale_states.npz")
    parser.add_argument("--target-generation", default="outputs/poc_v45_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--target-items", default="data/v45_hotpot_adapter_confirmation_candidates.jsonl")
    parser.add_argument("--output", default="outputs/poc_v45_frozen_cevr_adapter_confirmation_results.json")
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = load_npz(ROOT / args.train_cache)
    target_cache = load_npz(ROOT / args.target_cache)
    train = collapse_candidates(
        train_cache, ROOT / args.train_generation, "raw_L18", clean=True
    )
    target = collapse_candidates(
        target_cache, ROOT / args.target_generation, "raw_C_L18", clean=False
    )
    labels = target_cache["labels"].astype(int)
    qids = target_cache["question_ids"].astype(str)
    sample_indices = target_cache["sample_indices"].astype(int)

    first = first_choice(target_cache)
    likelihood = unique_choice(target, target.likelihood)
    original_risk = risk_scores(
        train_cache, target_cache, "raw_L18", "raw_C_L18"
    )
    original_linear = choose_per_question(qids, sample_indices, -original_risk)

    mask = mixed_question_mask(train)
    train_centered = center_by_question(train.x[mask], train.question_ids[mask])
    target_centered = center_by_question(target.x, target.question_ids)
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=4000, random_state=20260808
    )
    classifier.fit(scaler.fit_transform(train_centered), train.correct[mask])
    unique_linear_utility = classifier.predict_proba(
        scaler.transform(target_centered)
    )[:, 1]
    unique_linear = unique_choice(target, unique_linear_utility)

    utility, seed_predictions, parameter_count, _ = fit_ensemble(
        train, target, rank=32, centered=False, objective="listwise",
        epochs=100, device=device,
    )
    adapter = unique_choice(target, utility)
    seed_choices = [unique_choice(target, values) for values in seed_predictions]
    seed_agreement = float(np.mean([
        len({choice[qid] for choice in seed_choices}) == 1
        for qid in sorted(adapter)
    ]))

    choices = {
        "first_sample": first,
        "restricted_likelihood": likelihood,
        "original_linear_cevr": original_linear,
        "unique_centered_logistic": unique_linear,
        "frozen_rank32_listwise_adapter": adapter,
    }
    results = {}
    for name, choice in choices.items():
        result = accuracy_from_indices(labels, choice)
        if name != "first_sample":
            result["paired_vs_first"] = bootstrap_delta(
                labels, first, choice, 20260920
            )
        results[name] = result
    adapter_result = results["frozen_rank32_listwise_adapter"]
    adapter_result.update({
        "paired_vs_original_linear": bootstrap_delta(
            labels, original_linear, adapter, 20260921
        ),
        "paired_vs_likelihood": bootstrap_delta(
            labels, likelihood, adapter, 20260922
        ),
        "paired_vs_unique_centered_logistic": bootstrap_delta(
            labels, unique_linear, adapter, 20260923
        ),
        "candidate_population_auroc": float(
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                target.correct, utility
            )
        ),
        "within_question_auroc": within_question_auroc(
            target.correct, target.question_ids, utility
        ),
        "parameter_count": int(parameter_count),
        "three_seed_unanimous_choice_fraction": seed_agreement,
    })

    primary = adapter_result["paired_vs_original_linear"]
    secondary = adapter_result["paired_vs_likelihood"]
    t0 = target_cache["raw_T0_L18"].astype(np.float32)
    t0_max_difference = max(
        float(np.max(np.abs(t0[qids == qid] - t0[qids == qid][0])))
        for qid in np.unique(qids)
    )
    payload = {
        "experiment": "poc_v45_frozen_cevr_adapter_confirmation",
        "protocol": "paper/V45_FROZEN_CEVR_ADAPTER_CONFIRMATION_PROTOCOL.md",
        "status": "prospective source-ID-isolated confirmation; no V45 refit",
        "frozen_method": {
            "training_source": "V40f / 2WikiMultiHopQA",
            "representation": "unique sampled candidate; absolute raw residual L18",
            "adapter": "rank-32 linear-plus-GELU correction",
            "objective": "question-listwise correctness",
            "epochs": 100,
            "seeds": [20260808, 20260809, 20260810],
        },
        "target": {
            "dataset": "V45 / HotpotQA distractor validation",
            "n_rows": int(len(labels)),
            "n_questions": int(len(np.unique(qids))),
            "n_unique_sampled_candidates": int(len(target.correct)),
            "n_mixed_questions": int(len(np.unique(
                target.question_ids[mixed_question_mask(target)]
            ))),
            "reachable_oracle": coverage(labels, qids),
            "source_isolation_audit": source_isolation_audit(ROOT / args.target_items),
        },
        "t0_collision_control": {
            "max_within_question_state_difference": t0_max_difference,
            "within_question_auroc": 0.5,
            "selection_accuracy": results["first_sample"]["accuracy"],
            "decision": "identical candidate scores; earliest sample tie break",
        },
        "results": results,
        "decision": {
            "primary_delta_vs_original_linear": primary,
            "primary_pass": bool(primary["ci95"][0] > 0.0),
            "secondary_delta_vs_likelihood": secondary,
            "secondary_pass": bool(secondary["ci95"][0] > 0.0),
        },
        "claim_boundary": [
            "V45 is new for the adapter but remains within HotpotQA and Gemma-2-2B.",
            "The result concerns selection among sampled, evidence-grounded commitments.",
            "It does not establish future-branch prediction, calibrated refusal, causal support, or cross-model transfer.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "accuracies": {key: value["accuracy"] for key, value in results.items()},
        "adapter": adapter_result,
        "decision": payload["decision"],
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
