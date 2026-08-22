"""Frozen direct-text verifier baseline for the V45 CEVR candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import (  # noqa: E402
    accuracy_from_indices,
    bootstrap_delta,
    load_npz,
)
from scripts.poc_v44_framework_guided_cevr_finetuning import (  # noqa: E402
    collapse_candidates,
    first_choice,
    mixed_question_mask,
    unique_choice,
    within_question_auroc,
)


C_GRID = (0.1, 1.0, 10.0)
SEED = 20260822


def load_items(path: Path) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["item_id"]): row for row in rows}


def candidate_texts(data, items: dict[str, dict]) -> np.ndarray:
    texts = []
    for qid, option in zip(data.question_ids, data.options):
        row = items[str(qid)]
        candidate = str(row["options"][str(option)])
        texts.append(
            "[QUESTION] " + str(row["question"]) +
            " [EVIDENCE] " + str(row["facts"]) +
            " [CANDIDATE] " + candidate
        )
    return np.asarray(texts, dtype=object)


def build_model(c_value: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=30000,
            sublinear_tf=True, strip_accents="unicode",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=50000, sublinear_tf=True,
        )),
    ])
    classifier = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=4000,
        solver="liblinear", random_state=SEED,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def select_indices(data, utility: np.ndarray) -> dict[str, int]:
    return unique_choice(data, utility)


def selection_accuracy(data, utility: np.ndarray) -> float:
    selected = []
    for qid in np.unique(data.question_ids):
        indices = np.flatnonzero(data.question_ids == qid)
        order = np.lexsort((data.sample_indices[indices], -utility[indices]))
        selected.append(int(indices[order[0]]))
    selected = np.asarray(selected, dtype=int)
    return float(data.correct[selected].mean())


def source_cv(texts, data, mask: np.ndarray) -> tuple[float, dict]:
    x = texts[mask]
    y = data.correct[mask]
    groups = data.question_ids[mask]
    splitter = GroupKFold(n_splits=5)
    scores: dict[str, list[float]] = {str(c): [] for c in C_GRID}
    for c_value in C_GRID:
        for train_idx, valid_idx in splitter.split(x, y, groups):
            model = build_model(c_value)
            model.fit(x[train_idx], y[train_idx])
            utility = model.predict_proba(x[valid_idx])[:, 1]
            subset = type(data)(
                x=data.x[mask][valid_idx],
                correct=y[valid_idx],
                question_ids=groups[valid_idx],
                options=data.options[mask][valid_idx],
                likelihood=data.likelihood[mask][valid_idx],
                sample_indices=data.sample_indices[mask][valid_idx],
                source_indices=data.source_indices[mask][valid_idx],
            )
            scores[str(c_value)].append(selection_accuracy(subset, utility))
    means = {key: float(np.mean(values)) for key, values in scores.items()}
    selected_c = min(C_GRID, key=lambda c: (-means[str(c)], c))
    return float(selected_c), {"fold_scores": scores, "mean_accuracy": means}


def choices_from_record(record: dict, qids: np.ndarray) -> dict[str, int]:
    unique_qids = np.unique(qids.astype(str))
    indices = record["selected_indices"]
    if len(unique_qids) != len(indices):
        raise ValueError("Stored selection length does not match target questions")
    return {str(qid): int(index) for qid, index in zip(unique_qids, indices)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--train-generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--train-items", default="data/v40_machine_verifiable_candidates.jsonl")
    parser.add_argument("--target-cache", default="outputs/cache/v45_hotpot_no_rationale_states.npz")
    parser.add_argument("--target-generation", default="outputs/poc_v45_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--target-items", default="data/v45_hotpot_adapter_confirmation_candidates.jsonl")
    parser.add_argument("--cevr-record", default="outputs/poc_v45_frozen_cevr_adapter_confirmation_results.json")
    parser.add_argument("--output", default="outputs/poc_text_only_cevr_baseline_results.json")
    args = parser.parse_args()

    train_cache = load_npz(ROOT / args.train_cache)
    target_cache = load_npz(ROOT / args.target_cache)
    train = collapse_candidates(
        train_cache, ROOT / args.train_generation, "raw_L18", clean=True
    )
    target = collapse_candidates(
        target_cache, ROOT / args.target_generation, "raw_C_L18", clean=False
    )
    train_text = candidate_texts(train, load_items(ROOT / args.train_items))
    target_text = candidate_texts(target, load_items(ROOT / args.target_items))
    train_mask = mixed_question_mask(train)
    selected_c, cv = source_cv(train_text, train, train_mask)

    model = build_model(selected_c)
    model.fit(train_text[train_mask], train.correct[train_mask])
    target_utility = model.predict_proba(target_text)[:, 1]
    text_choice = select_indices(target, target_utility)
    labels = target_cache["labels"].astype(int)
    target_qids = target_cache["question_ids"].astype(str)
    text_result = accuracy_from_indices(labels, text_choice)

    first = first_choice(target_cache)
    likelihood = unique_choice(target, target.likelihood)
    frozen = json.loads((ROOT / args.cevr_record).read_text(encoding="utf-8"))
    dense_unique = choices_from_record(
        frozen["results"]["unique_centered_logistic"], target_qids
    )
    dense_listwise = choices_from_record(
        frozen["results"]["frozen_rank32_listwise_adapter"], target_qids
    )
    text_result.update({
        "paired_vs_first": bootstrap_delta(labels, first, text_choice, SEED + 1),
        "paired_vs_likelihood": bootstrap_delta(labels, likelihood, text_choice, SEED + 2),
        "paired_vs_dense_unique": bootstrap_delta(labels, dense_unique, text_choice, SEED + 3),
        "paired_vs_dense_listwise": bootstrap_delta(labels, dense_listwise, text_choice, SEED + 4),
        "candidate_population_auroc": float(roc_auc_score(target.correct, target_utility)),
        "candidate_population_auprc": float(average_precision_score(target.correct, target_utility)),
        "within_question_auroc": within_question_auroc(
            target.correct, target.question_ids, target_utility
        ),
    })

    payload = {
        "experiment": "poc_text_only_cevr_baseline",
        "protocol": "paper/TEXT_ONLY_VERIFIER_BASELINE_PROTOCOL_2026-08-22.md",
        "status": "frozen-before-run retrospective target comparison; no V45 tuning",
        "interface": "question + evidence + candidate text only",
        "source": {
            "n_questions": int(len(np.unique(train.question_ids))),
            "n_unique_candidates": int(len(train.correct)),
            "n_mixed_questions": int(len(np.unique(train.question_ids[train_mask]))),
            "n_fit_candidates": int(train_mask.sum()),
            "selected_C": selected_c,
            "grouped_cv": cv,
        },
        "target": {
            "n_questions": int(len(np.unique(target.question_ids))),
            "n_unique_candidates": int(len(target.correct)),
        },
        "result": text_result,
        "claim_boundary": [
            "This is a direct TF-IDF text verifier, not a pretrained Transformer cross-encoder.",
            "Target labels were used only for final evaluation and paired intervals.",
            "The result cannot establish superiority over all textual evidence verifiers.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
