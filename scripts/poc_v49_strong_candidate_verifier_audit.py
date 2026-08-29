"""V49: task-aligned strong verifier audit on the V48 500-question target."""

from __future__ import annotations

import argparse
import gc
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260827
N_BOOTSTRAP = 10_000
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LARGE_MODEL = str(ROOT / "models" / "Qwen2.5-7B-Instruct")
C_GRID = (0.1, 1.0, 10.0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_rows(items: list[dict]) -> dict[str, np.ndarray]:
    prompts, qids, labels, order = [], [], [], []
    for item in items:
        options = item["options"]
        for position, option in enumerate(("A", "B", "C", "D")):
            candidate = str(options[option])
            prompts.append(
                "[QUESTION] " + str(item["question"]) +
                "\n[EVIDENCE] " + str(item["facts"]) +
                "\n[CANDIDATE] " + candidate +
                "\n[TASK] Is the candidate supported by the evidence? Verdict:"
            )
            qids.append(str(item["item_id"]))
            labels.append(int(option == item["gold_option"]))
            order.append(position)
    return {
        "prompts": np.asarray(prompts, dtype=object),
        "qids": np.asarray(qids, dtype=object),
        "correct": np.asarray(labels, dtype=np.int8),
        "order": np.asarray(order, dtype=np.int8),
    }


def select(qids: np.ndarray, order: np.ndarray, utility: np.ndarray) -> np.ndarray:
    output = []
    for qid in np.unique(qids.astype(str)):
        indices = np.flatnonzero(qids.astype(str) == qid)
        ranking = np.lexsort((order[indices], -utility[indices]))
        output.append(int(indices[ranking[0]]))
    return np.asarray(output, dtype=int)


def per_question_auc(correct: np.ndarray, qids: np.ndarray, utility: np.ndarray) -> np.ndarray:
    values = []
    for qid in np.unique(qids.astype(str)):
        indices = np.flatnonzero(qids.astype(str) == qid)
        values.append(float(roc_auc_score(correct[indices], utility[indices])))
    return np.asarray(values)


def bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "n_questions": int(len(values)),
    }


def metrics(data: dict[str, np.ndarray], utility: np.ndarray, seed: int) -> dict:
    chosen = select(data["qids"], data["order"], utility)
    return {
        "within_question_auroc": bootstrap(per_question_auc(data["correct"], data["qids"], utility), seed),
        "selection_accuracy": bootstrap(data["correct"][chosen], seed + 1),
        "population_auroc": float(roc_auc_score(data["correct"], utility)),
        "population_auprc": float(average_precision_score(data["correct"], utility)),
        "selected_indices": [int(i) for i in chosen],
    }


def paired_delta(a: dict, b: dict, data: dict[str, np.ndarray], seed: int) -> dict:
    chosen_a = np.asarray(a["selected_indices"], dtype=int)
    chosen_b = np.asarray(b["selected_indices"], dtype=int)
    values = data["correct"][chosen_a] - data["correct"][chosen_b]
    return bootstrap(values, seed)


def tokenize(tokenizer, prompts: np.ndarray, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(list(prompts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")


@torch.inference_mode()
def encode_frozen(model_id: str, prompts: np.ndarray, batch_size: int, max_length: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_id, local_files_only=True, torch_dtype=dtype).to(device).eval()
    result = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="V49 frozen text encoder"):
        batch = tokenizer(list(prompts[start:start + batch_size]), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        state = model(**batch, return_dict=True).last_hidden_state
        last = batch["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(last), device=device)
        result.append(state[rows, last].float().cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(result).astype(np.float32)


def fit_utility(train_x: np.ndarray, train_y: np.ndarray, target_x: np.ndarray, c_value: float) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=4000, random_state=SEED)
    model.fit(scaler.transform(train_x), train_y)
    return model.predict_proba(scaler.transform(target_x))[:, 1]


def frozen_encoder_baseline(source: dict, target: dict, args) -> dict:
    features = encode_frozen(args.small_model, np.concatenate([source["prompts"], target["prompts"]]), args.batch_size, args.max_length)
    source_x, target_x = features[:len(source["correct"])], features[len(source["correct"]) :]
    scores = {str(c): [] for c in C_GRID}
    group_cv = GroupKFold(n_splits=5)
    for c_value in C_GRID:
        for tr, va in group_cv.split(source_x, source["correct"], source["qids"]):
            utility = fit_utility(source_x[tr], source["correct"][tr], source_x[va], c_value)
            fold = {key: value[va] for key, value in source.items() if key != "prompts"}
            scores[str(c_value)].append(metrics(fold, utility, SEED)["selection_accuracy"]["mean"])
    mean_scores = {key: float(np.mean(value)) for key, value in scores.items()}
    selected_c = min(C_GRID, key=lambda c: (-mean_scores[str(c)], c))
    utility = fit_utility(source_x, source["correct"], target_x, selected_c)
    result = metrics(target, utility, SEED + 10)
    return {
        "interface": "frozen Qwen2.5-0.5B text representation + source-fit logistic head",
        "source_cv": {"selected_C": selected_c, "selection_accuracy": mean_scores},
        "target": result,
    }


def amp_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def load_classifier(model_id: str, pad_id: int, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, local_files_only=True, torch_dtype=torch.float32)
    model.config.pad_token_id = pad_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model.to(device)


def train_epochs(model, tokens, labels, epochs, batch_size, accumulation, device, callback=None):
    dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    counts = torch.bincount(labels, minlength=2).float()
    weights = (counts.sum() / (2 * counts.clamp_min(1))).to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (ids, mask, y) in enumerate(tqdm(loader, desc=f"V49 text fine-tune {epoch}"), start=1):
            with amp_context(device):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
            loss = F.cross_entropy(logits, y.to(device), weight=weights) / accumulation
            loss.backward()
            total_loss += float(loss.item()) * accumulation
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "mean_train_loss": total_loss / len(loader)}
        if callback:
            row.update(callback(model))
        history.append(row)
    return history


@torch.inference_mode()
def classifier_utility(model, tokens, batch_size, device):
    model.eval()
    values = []
    for start in range(0, tokens["input_ids"].shape[0], batch_size):
        batch = {key: value[start:start + batch_size].to(device) for key, value in tokens.items()}
        with amp_context(device):
            logits = model(**batch).logits.float()
        values.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def finetuned_encoder_baseline(source: dict, target: dict, args) -> dict:
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.small_model, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    source_tokens = tokenize(tokenizer, source["prompts"], args.max_length)
    target_tokens = tokenize(tokenizer, target["prompts"], args.max_length)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit, valid = next(split.split(source["correct"], groups=source["qids"]))
    fit_tokens = {key: value[fit] for key, value in source_tokens.items()}
    valid_tokens = {key: value[valid] for key, value in source_tokens.items()}
    valid_data = {key: value[valid] for key, value in source.items() if key != "prompts"}
    model = load_classifier(args.small_model, tokenizer.pad_token_id, device)
    def callback(current):
        return {"validation_selection_accuracy": metrics(valid_data, classifier_utility(current, valid_tokens, args.batch_size, device), SEED)["selection_accuracy"]["mean"]}
    history = train_epochs(model, fit_tokens, torch.as_tensor(source["correct"][fit], dtype=torch.long), 3, args.batch_size, args.gradient_accumulation, device, callback)
    epochs = min(history, key=lambda r: (-r["validation_selection_accuracy"], r["epoch"]))["epoch"]
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_all(SEED)
    model = load_classifier(args.small_model, tokenizer.pad_token_id, device)
    final_history = train_epochs(model, source_tokens, torch.as_tensor(source["correct"], dtype=torch.long), epochs, args.batch_size, args.gradient_accumulation, device)
    result = metrics(target, classifier_utility(model, target_tokens, args.batch_size, device), SEED + 20)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "interface": "source-tuned Qwen2.5-0.5B text cross-encoder",
        "epoch_selection": history,
        "selected_epochs": int(epochs),
        "final_training": final_history,
        "target": result,
    }


def zero_shot_7b_baseline(target: dict, args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.large_model, local_files_only=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.large_model, local_files_only=True, device_map="auto", quantization_config=quant, torch_dtype=torch.bfloat16).eval()
    sequences, boundaries = [], []
    for prompt in target["prompts"]:
        prefix = tokenizer(str(prompt), add_special_tokens=True)["input_ids"]
        for continuation in (" supported", " unsupported"):
            suffix = tokenizer(continuation, add_special_tokens=False)["input_ids"]
            ids = (prefix + suffix)[-args.max_length:]
            sequences.append(ids)
            boundaries.append(max(1, len(ids) - len(suffix)))
    scores = []
    for start in tqdm(range(0, len(sequences), 2), desc="V49 zero-shot Qwen-7B verifier"):
        batch_sequences = sequences[start:start + 2]
        batch = tokenizer.pad({"input_ids": batch_sequences}, padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        for row, ids in enumerate(batch_sequences):
            boundary = boundaries[start + row]
            suffix = ids[boundary:]
            scores.append(float(sum(logp[row, boundary + j - 1, token].item() for j, token in enumerate(suffix))))
    utilities = np.asarray(scores[0::2]) - np.asarray(scores[1::2])
    del model
    torch.cuda.empty_cache()
    return {
        "interface": "zero-shot Qwen2.5-7B-Instruct NF4 supported-versus-unsupported likelihood ratio",
        "target": metrics(target, utilities, SEED + 30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-items", default="data/v40_machine_verifiable_candidates.jsonl")
    parser.add_argument("--target-items", default="data/v47_hotpot_scale_confirmation_candidates.jsonl")
    parser.add_argument("--small-model", default=SMALL_MODEL)
    parser.add_argument("--large-model", default=LARGE_MODEL)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=("frozen", "finetuned", "zero-shot-7b"), default=("frozen", "finetuned", "zero-shot-7b"))
    parser.add_argument("--output", default="outputs/poc_v49_strong_candidate_verifier_audit.json")
    args = parser.parse_args()
    source = candidate_rows(load_items(ROOT / args.source_items))
    target = candidate_rows(load_items(ROOT / args.target_items))
    payload = {
        "experiment": "V49 strong candidate-verifier audit",
        "protocol": "paper/V49_STRONG_CANDIDATE_VERIFIER_AUDIT_PROTOCOL.md",
        "status": "post-hoc robustness audit; no V48 target refit or hyperparameter selection",
        "source": {"n_questions": int(len(np.unique(source["qids"]))), "n_candidates": int(len(source["correct"]))},
        "target": {"n_questions": int(len(np.unique(target["qids"]))), "n_candidates": int(len(target["correct"]))},
        "methods": {},
        "claim_boundary": "Task-aligned candidate-verifier controls. This is not a reproduction of semantic-entropy or SelfCheckGPT, whose primary endpoint is sampled-answer risk rather than selection among a supplied candidate set.",
    }
    if "frozen" in args.methods:
        payload["methods"]["frozen_qwen_0p5b_cross_encoder"] = frozen_encoder_baseline(source, target, args)
    if "finetuned" in args.methods:
        payload["methods"]["source_tuned_qwen_0p5b_cross_encoder"] = finetuned_encoder_baseline(source, target, args)
    if "zero-shot-7b" in args.methods:
        payload["methods"]["zero_shot_qwen_7b_verifier"] = zero_shot_7b_baseline(target, args)
    (ROOT / args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
