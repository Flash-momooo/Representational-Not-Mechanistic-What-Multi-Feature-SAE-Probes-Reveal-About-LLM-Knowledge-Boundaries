"""Frozen Qwen-0.5B text cross-encoder baseline for the V45 candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import (  # noqa: E402
    accuracy_from_indices,
    bootstrap_delta,
    load_npz,
)
from scripts.poc_text_only_cevr_baseline import (  # noqa: E402
    candidate_texts,
    choices_from_record,
    load_items,
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
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def verifier_prompts(texts: np.ndarray) -> list[str]:
    return [
        str(text) +
        " [TASK] Determine whether the candidate is supported by the evidence. Answer:"
        for text in texts
    ]


@torch.inference_mode()
def encode(prompts: list[str], model_id: str, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_id, local_files_only=True, dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="text encoder"):
        batch = tokenizer(
            prompts[start:start + batch_size], padding=True, truncation=True,
            max_length=1536, return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        hidden = model(**batch, return_dict=True).last_hidden_state
        last = batch["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(hidden.shape[0], device=device)
        features.append(hidden[rows, last].float().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


def select_local(data, utility: np.ndarray) -> np.ndarray:
    selected = []
    for qid in np.unique(data.question_ids):
        indices = np.flatnonzero(data.question_ids == qid)
        order = np.lexsort((data.sample_indices[indices], -utility[indices]))
        selected.append(int(indices[order[0]]))
    return np.asarray(selected, dtype=int)


def fit_predict(train_x, train_y, target_x, c_value: float):
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    target_z = scaler.transform(target_x)
    model = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=4000,
        random_state=SEED,
    )
    model.fit(train_z, train_y)
    return model.predict_proba(target_z)[:, 1]


def source_cv(features, data, mask: np.ndarray) -> tuple[float, dict]:
    x = features[mask]
    y = data.correct[mask]
    groups = data.question_ids[mask]
    splitter = GroupKFold(n_splits=5)
    scores = {str(c): [] for c in C_GRID}
    for c_value in C_GRID:
        for train_idx, valid_idx in splitter.split(x, y, groups):
            utility = fit_predict(
                x[train_idx], y[train_idx], x[valid_idx], c_value
            )
            subset = type(data)(
                x=data.x[mask][valid_idx], correct=y[valid_idx],
                question_ids=groups[valid_idx],
                options=data.options[mask][valid_idx],
                likelihood=data.likelihood[mask][valid_idx],
                sample_indices=data.sample_indices[mask][valid_idx],
                source_indices=data.source_indices[mask][valid_idx],
            )
            chosen = select_local(subset, utility)
            scores[str(c_value)].append(float(subset.correct[chosen].mean()))
    means = {key: float(np.mean(values)) for key, values in scores.items()}
    selected_c = min(C_GRID, key=lambda c: (-means[str(c)], c))
    return float(selected_c), {"fold_scores": scores, "mean_accuracy": means}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--train-generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--train-items", default="data/v40_machine_verifiable_candidates.jsonl")
    parser.add_argument("--target-cache", default="outputs/cache/v45_hotpot_no_rationale_states.npz")
    parser.add_argument("--target-generation", default="outputs/poc_v45_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--target-items", default="data/v45_hotpot_adapter_confirmation_candidates.jsonl")
    parser.add_argument("--cevr-record", default="outputs/poc_v45_frozen_cevr_adapter_confirmation_results.json")
    parser.add_argument("--tfidf-record", default="outputs/poc_text_only_cevr_baseline_results.json")
    parser.add_argument("--output", default="outputs/poc_frozen_text_cross_encoder_results.json")
    args = parser.parse_args()

    train_cache = load_npz(ROOT / args.train_cache)
    target_cache = load_npz(ROOT / args.target_cache)
    train = collapse_candidates(
        train_cache, ROOT / args.train_generation, "raw_L18", clean=True
    )
    target = collapse_candidates(
        target_cache, ROOT / args.target_generation, "raw_C_L18", clean=False
    )
    train_prompts = verifier_prompts(candidate_texts(
        train, load_items(ROOT / args.train_items)
    ))
    target_prompts = verifier_prompts(candidate_texts(
        target, load_items(ROOT / args.target_items)
    ))
    all_features = encode(train_prompts + target_prompts, args.model, args.batch_size)
    train_features = all_features[:len(train_prompts)]
    target_features = all_features[len(train_prompts):]
    train_mask = mixed_question_mask(train)
    selected_c, cv = source_cv(train_features, train, train_mask)
    target_utility = fit_predict(
        train_features[train_mask], train.correct[train_mask],
        target_features, selected_c,
    )
    text_choice = unique_choice(target, target_utility)

    labels = target_cache["labels"].astype(int)
    qids = target_cache["question_ids"].astype(str)
    first = first_choice(target_cache)
    likelihood = unique_choice(target, target.likelihood)
    frozen = json.loads((ROOT / args.cevr_record).read_text(encoding="utf-8"))
    dense_unique = choices_from_record(frozen["results"]["unique_centered_logistic"], qids)
    dense_listwise = choices_from_record(
        frozen["results"]["frozen_rank32_listwise_adapter"], qids
    )
    tfidf = json.loads((ROOT / args.tfidf_record).read_text(encoding="utf-8"))
    tfidf_choice = choices_from_record(tfidf["result"], qids)

    from sklearn.metrics import average_precision_score, roc_auc_score
    result = accuracy_from_indices(labels, text_choice)
    result.update({
        "paired_vs_first": bootstrap_delta(labels, first, text_choice, SEED + 11),
        "paired_vs_likelihood": bootstrap_delta(labels, likelihood, text_choice, SEED + 12),
        "paired_vs_tfidf": bootstrap_delta(labels, tfidf_choice, text_choice, SEED + 13),
        "paired_vs_dense_unique": bootstrap_delta(labels, dense_unique, text_choice, SEED + 14),
        "paired_vs_dense_listwise": bootstrap_delta(labels, dense_listwise, text_choice, SEED + 15),
        "candidate_population_auroc": float(roc_auc_score(target.correct, target_utility)),
        "candidate_population_auprc": float(average_precision_score(target.correct, target_utility)),
        "within_question_auroc": within_question_auroc(
            target.correct, target.question_ids, target_utility
        ),
    })
    payload = {
        "experiment": "poc_frozen_text_cross_encoder",
        "protocol": "paper/FROZEN_TEXT_CROSS_ENCODER_PROTOCOL_2026-08-22.md",
        "status": "retrospective fairness audit; no target refit",
        "encoder": args.model,
        "encoder_trainable": False,
        "source": {
            "n_questions": int(len(np.unique(train.question_ids))),
            "n_unique_candidates": int(len(train.correct)),
            "n_mixed_questions": int(len(np.unique(train.question_ids[train_mask]))),
            "selected_C": selected_c,
            "grouped_cv": cv,
        },
        "target": {
            "n_questions": int(len(np.unique(target.question_ids))),
            "n_unique_candidates": int(len(target.correct)),
        },
        "result": result,
        "claim_boundary": [
            "Frozen 0.5B encoder plus supervised logistic head; not end-to-end NLI fine-tuning.",
            "Same realized option pool and source supervision as CEVR; no V45 tuning.",
            "Retrospective fairness audit, not a new prospective confirmation.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
