"""Source-tuned Qwen-0.5B text cross-encoder baseline for V45 candidates."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import (  # noqa: E402
    accuracy_from_indices,
    bootstrap_delta,
    load_npz,
)
from scripts.poc_frozen_text_cross_encoder import select_local, verifier_prompts
from scripts.poc_text_only_cevr_baseline import (
    candidate_texts,
    choices_from_record,
    load_items,
)
from scripts.poc_v44_framework_guided_cevr_finetuning import (
    collapse_candidates,
    first_choice,
    mixed_question_mask,
    unique_choice,
    within_question_auroc,
)


SEED = 20260822
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        prompts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )


def load_model(model_id: str, pad_token_id: int, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, local_files_only=True, dtype=torch.float32,
    )
    model.config.pad_token_id = pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model.to(device)


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def predict(model, tokens: dict[str, torch.Tensor], batch_size: int, device):
    model.eval()
    values = []
    n = tokens["input_ids"].shape[0]
    for start in range(0, n, batch_size):
        batch = {
            key: value[start:start + batch_size].to(device)
            for key, value in tokens.items()
        }
        with amp_context(device):
            logits = model(**batch).logits.float()
        values.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def train_epochs(
    model, tokens, labels, epochs: int, batch_size: int,
    accumulation: int, device, eval_callback=None,
):
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    counts = torch.bincount(labels, minlength=2).float()
    weights = (counts.sum() / (2 * counts.clamp_min(1))).to(device)
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (input_ids, attention_mask, y) in enumerate(
            tqdm(loader, desc=f"fine-tune epoch {epoch}"), start=1
        ):
            with amp_context(device):
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                ).logits.float()
            loss = F.cross_entropy(logits, y.to(device), weight=weights) / accumulation
            loss.backward()
            total_loss += float(loss.item()) * accumulation
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        record = {"epoch": epoch, "mean_train_loss": total_loss / len(loader)}
        if eval_callback is not None:
            record.update(eval_callback(model))
        history.append(record)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--train-generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--train-items", default="data/v40_machine_verifiable_candidates.jsonl")
    parser.add_argument("--target-cache", default="outputs/cache/v45_hotpot_no_rationale_states.npz")
    parser.add_argument("--target-generation", default="outputs/poc_v45_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--target-items", default="data/v45_hotpot_adapter_confirmation_candidates.jsonl")
    parser.add_argument("--cevr-record", default="outputs/poc_v45_frozen_cevr_adapter_confirmation_results.json")
    parser.add_argument("--output", default="outputs/poc_finetuned_text_cross_encoder_results.json")
    args = parser.parse_args()

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = load_npz(ROOT / args.train_cache)
    target_cache = load_npz(ROOT / args.target_cache)
    train = collapse_candidates(train_cache, ROOT / args.train_generation, "raw_L18", clean=True)
    target = collapse_candidates(target_cache, ROOT / args.target_generation, "raw_C_L18", clean=False)
    train_mask = mixed_question_mask(train)
    train_indices = np.flatnonzero(train_mask)

    train_prompts_all = verifier_prompts(candidate_texts(train, load_items(ROOT / args.train_items)))
    target_prompts = verifier_prompts(candidate_texts(target, load_items(ROOT / args.target_items)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tokens_all = tokenize(tokenizer, train_prompts_all, args.max_length)
    target_tokens = tokenize(tokenizer, target_prompts, args.max_length)

    groups = train.question_ids[train_indices]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit_local, valid_local = next(splitter.split(train_indices, groups=groups))
    fit_indices = train_indices[fit_local]
    valid_indices = train_indices[valid_local]
    fit_tokens = {key: value[fit_indices] for key, value in train_tokens_all.items()}
    valid_tokens = {key: value[valid_indices] for key, value in train_tokens_all.items()}
    fit_labels = torch.as_tensor(train.correct[fit_indices], dtype=torch.long)

    model = load_model(args.model, tokenizer.pad_token_id, device)
    valid_subset = type(train)(
        x=train.x[valid_indices], correct=train.correct[valid_indices],
        question_ids=train.question_ids[valid_indices], options=train.options[valid_indices],
        likelihood=train.likelihood[valid_indices],
        sample_indices=train.sample_indices[valid_indices],
        source_indices=train.source_indices[valid_indices],
    )

    def evaluate_validation(current_model):
        utility = predict(current_model, valid_tokens, args.batch_size, device)
        selected = select_local(valid_subset, utility)
        return {"validation_selection_accuracy": float(valid_subset.correct[selected].mean())}

    selection_history = train_epochs(
        model, fit_tokens, fit_labels, 3, args.batch_size,
        args.gradient_accumulation, device, evaluate_validation,
    )
    best_epoch = min(
        selection_history,
        key=lambda row: (-row["validation_selection_accuracy"], row["epoch"]),
    )["epoch"]
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_all(SEED)
    model = load_model(args.model, tokenizer.pad_token_id, device)
    all_fit_tokens = {key: value[train_indices] for key, value in train_tokens_all.items()}
    all_fit_labels = torch.as_tensor(train.correct[train_indices], dtype=torch.long)
    final_history = train_epochs(
        model, all_fit_tokens, all_fit_labels, int(best_epoch), args.batch_size,
        args.gradient_accumulation, device,
    )
    target_utility = predict(model, target_tokens, args.batch_size, device)
    text_choice = unique_choice(target, target_utility)

    labels = target_cache["labels"].astype(int)
    qids = target_cache["question_ids"].astype(str)
    first = first_choice(target_cache)
    likelihood = unique_choice(target, target.likelihood)
    frozen = json.loads((ROOT / args.cevr_record).read_text(encoding="utf-8"))
    dense_unique = choices_from_record(frozen["results"]["unique_centered_logistic"], qids)
    dense_listwise = choices_from_record(frozen["results"]["frozen_rank32_listwise_adapter"], qids)

    from sklearn.metrics import average_precision_score, roc_auc_score
    result = accuracy_from_indices(labels, text_choice)
    result.update({
        "paired_vs_first": bootstrap_delta(labels, first, text_choice, SEED + 21),
        "paired_vs_likelihood": bootstrap_delta(labels, likelihood, text_choice, SEED + 22),
        "paired_vs_dense_unique": bootstrap_delta(labels, dense_unique, text_choice, SEED + 23),
        "paired_vs_dense_listwise": bootstrap_delta(labels, dense_listwise, text_choice, SEED + 24),
        "candidate_population_auroc": float(roc_auc_score(target.correct, target_utility)),
        "candidate_population_auprc": float(average_precision_score(target.correct, target_utility)),
        "within_question_auroc": within_question_auroc(target.correct, target.question_ids, target_utility),
    })
    payload = {
        "experiment": "poc_finetuned_text_cross_encoder",
        "protocol": "paper/FINETUNED_TEXT_CROSS_ENCODER_PROTOCOL_2026-08-22.md",
        "status": "retrospective source-tuned fairness audit; no target refit",
        "model": args.model,
        "trainable": "all encoder and classification-head parameters",
        "source": {
            "n_questions": int(len(np.unique(train.question_ids))),
            "n_unique_candidates": int(len(train.correct)),
            "n_mixed_questions": int(len(np.unique(train.question_ids[train_mask]))),
            "n_fit_candidates": int(train_mask.sum()),
            "selection_history": selection_history,
            "selected_epochs": int(best_epoch),
            "final_history": final_history,
        },
        "target": {
            "n_questions": int(len(np.unique(target.question_ids))),
            "n_unique_candidates": int(len(target.correct)),
        },
        "result": result,
        "claim_boundary": [
            "Source-tuned 0.5B text cross-encoder; target labels used only for evaluation.",
            "Retrospective fairness audit, not prospective confirmation.",
            "Does not exhaust larger NLI verifiers or target-domain supervision.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
