"""Collect frozen FPE5 trajectories and L18 dense/SAE states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v37_same_question_trajectory import (
    clean_generated_ids,
    prefix_hash,
    score_answer,
)
from src.extract import ResidualHook
from src.load import load_all


LAYER = 18
STAGES = tuple(range(6))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_questions(dataset: str, n_questions: int) -> list[dict]:
    if dataset == "hotpot":
        source = read_jsonl(ROOT / "data" / "v41_hotpot_confirmatory_candidates.jsonl")
        output = []
        for item in source[:n_questions]:
            prompt = (
                "Answer using only the evidence below. Return only the shortest exact "
                "answer span copied verbatim from the evidence. Do not explain or "
                "rephrase.\n"
                f"Evidence:\n{item['facts']}\nQuestion: {item['question']}\nAnswer:"
            )
            output.append({
                "question_id": item["item_id"],
                "pair_id": item["pair_id"],
                "difficulty": item["level"],
                "prompt": prompt,
                "gold_answer": item["gold_answer"],
            })
        return output
    if dataset == "2wiki":
        existing = read_jsonl(ROOT / "data" / "v37_2wiki_trajectory.jsonl")
        by_question = {}
        for row in existing:
            by_question.setdefault(row["question_id"], {
                "question_id": row["question_id"],
                "pair_id": row["pair_id"],
                "difficulty": row["difficulty"],
                "prompt": row["prompt"],
                "gold_answer": row["gold_answer"],
            })
        return list(by_question.values())[:n_questions]
    raise ValueError(dataset)


def format_prompt(tokenizer, prompt: str, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_rows(
    assets,
    questions: list[dict],
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    chat_template: bool,
) -> list[dict]:
    rows = []
    stop_ids = {int(assets.tokenizer.eos_token_id)}
    for question_index, question in enumerate(tqdm(questions, desc="fpe5-generate")):
        prompt = format_prompt(assets.tokenizer, question["prompt"], chat_template)
        inputs = assets.tokenizer(prompt, return_tensors="pt").to(assets.device)
        torch.manual_seed(seed + question_index)
        with torch.no_grad():
            generated = assets.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=n_samples,
                max_new_tokens=max_new_tokens,
                eos_token_id=sorted(stop_ids),
                pad_token_id=assets.tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        for sample_index, sequence in enumerate(generated):
            token_ids = clean_generated_ids(
                sequence[prompt_length:].detach().cpu().tolist(), stop_ids
            )
            decoded = assets.tokenizer.decode(token_ids, skip_special_tokens=True)
            nonempty = [line.strip() for line in decoded.splitlines() if line.strip()]
            response = nonempty[0] if nonempty else decoded.strip()
            if response.lower().startswith("answer:"):
                response = response.split(":", 1)[1].strip()
            correct, exact, f1 = score_answer(response, question["gold_answer"])
            rows.append({
                **question,
                "prompt": prompt,
                "sample_index": sample_index,
                "model_answer": response,
                "generated_token_ids": token_ids,
                "model_correct": correct,
                "normalized_exact_match": exact,
                "token_f1": f1,
            })
    return rows


def distribution_features(logits: torch.Tensor) -> tuple[float, float, float]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = torch.exp(log_probs)
    top2 = torch.topk(probabilities, k=2).values
    return (
        float(top2[0].detach().cpu()),
        float((-(probabilities * log_probs).sum()).detach().cpu()),
        float((top2[0] - top2[1]).detach().cpu()),
    )


def extract_states(assets, rows: list[dict], cache_path: Path) -> dict[str, np.ndarray]:
    hook = ResidualHook().attach(assets.model.model.layers[LAYER])
    n_rows = len(rows)
    hidden_size = int(assets.model.config.hidden_size)
    d_sae = int(assets.saes[LAYER].cfg.d_sae)
    valid = {stage: np.zeros(n_rows, dtype=bool) for stage in STAGES}
    confidence = {stage: np.zeros((n_rows, 5), dtype=np.float32) for stage in STAGES}
    token_prefix = {stage: np.zeros((n_rows, 1024), dtype=np.float32) for stage in STAGES}
    raw = {
        stage: np.zeros((n_rows, hidden_size), dtype=np.float16) for stage in STAGES
    }
    sae = {
        stage: np.zeros((n_rows, d_sae), dtype=np.float16) for stage in STAGES
    }
    generated = [list(map(int, row["generated_token_ids"])) for row in rows]
    selected_log_probs = [[] for _ in rows]
    grouped = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[row["question_id"]].append(row_index)
        for stage in STAGES:
            valid[stage][row_index] = stage == 0 or len(generated[row_index]) >= stage
            token_prefix[stage][row_index] = prefix_hash(generated[row_index], stage)

    try:
        for _, row_indices in tqdm(grouped.items(), desc="fpe5-extract"):
            prompt_ids = assets.tokenizer(
                rows[row_indices[0]]["prompt"], return_tensors="pt"
            )["input_ids"].to(assets.device)
            for stage in STAGES:
                active = [index for index in row_indices if valid[stage][index]]
                if not active:
                    continue
                if stage == 0:
                    input_ids = prompt_ids
                else:
                    prefixes = torch.tensor(
                        [generated[index][:stage] for index in active],
                        dtype=prompt_ids.dtype,
                        device=assets.device,
                    )
                    input_ids = torch.cat(
                        [prompt_ids.expand(len(active), -1), prefixes], dim=1
                    )
                with torch.no_grad():
                    output = assets.model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                    )
                logits = output.logits[:, -1]
                vectors = hook.value[:, -1]
                with torch.no_grad():
                    latents = assets.saes[LAYER].encode(vectors)
                raw_values = vectors.float().cpu().numpy().astype(np.float16)
                sae_values = latents.float().cpu().numpy().astype(np.float16)

                for batch_index, row_index in enumerate(active):
                    source_index = 0 if stage == 0 else batch_index
                    max_prob, entropy, margin = distribution_features(logits[source_index])
                    history = selected_log_probs[row_index]
                    confidence[stage][row_index] = np.asarray([
                        max_prob,
                        entropy,
                        margin,
                        float(np.mean(history)) if history else 0.0,
                        float(np.min(history)) if history else 0.0,
                    ], dtype=np.float32)
                    raw[stage][row_index] = raw_values[source_index]
                    sae[stage][row_index] = sae_values[source_index]
                    if len(generated[row_index]) > stage:
                        next_token = generated[row_index][stage]
                        log_prob = torch.log_softmax(logits[source_index].float(), -1)[next_token]
                        selected_log_probs[row_index].append(float(log_prob.detach().cpu()))
    finally:
        hook.detach()

    payload = {
        "labels": np.asarray([0 if row["model_correct"] else 1 for row in rows], dtype=np.int8),
        "question_ids": np.asarray([row["question_id"] for row in rows], dtype=object),
        "pair_ids": np.asarray([row["pair_id"] for row in rows], dtype=object),
        "difficulty": np.asarray([row["difficulty"] for row in rows], dtype=object),
        "answer_lengths": np.asarray([len(row["generated_token_ids"]) for row in rows], dtype=np.int16),
    }
    for stage in STAGES:
        payload[f"valid_T{stage}"] = valid[stage]
        payload[f"confidence_T{stage}"] = confidence[stage]
        payload[f"token_prefix_T{stage}"] = token_prefix[stage]
        payload[f"raw_T{stage}_L{LAYER}"] = raw[stage]
        payload[f"sae_T{stage}_L{LAYER}"] = sae[stage]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def reconstruction_audit(assets, states: dict[str, np.ndarray]) -> dict:
    raw_blocks = []
    sae_blocks = []
    for stage in STAGES:
        valid = states[f"valid_T{stage}"].astype(bool)
        raw_blocks.append(states[f"raw_T{stage}_L{LAYER}"][valid].astype(np.float32))
        sae_blocks.append(states[f"sae_T{stage}_L{LAYER}"][valid].astype(np.float32))
    raw = np.vstack(raw_blocks)
    latent = np.vstack(sae_blocks)
    reconstructed = []
    for start in range(0, len(latent), 128):
        batch = torch.as_tensor(
            latent[start:start + 128], dtype=torch.bfloat16, device=assets.device
        )
        with torch.no_grad():
            decoded = assets.saes[LAYER].decode(batch)
        reconstructed.append(decoded.float().cpu().numpy())
    reconstruction = np.vstack(reconstructed)
    error = raw - reconstruction
    centered = raw - raw.mean(axis=0, keepdims=True)
    explained_variance = 1.0 - float(np.sum(error ** 2) / np.sum(centered ** 2))
    cosine = np.sum(raw * reconstruction, axis=1) / np.maximum(
        np.linalg.norm(raw, axis=1) * np.linalg.norm(reconstruction, axis=1), 1e-12
    )
    l0 = np.count_nonzero(latent, axis=1)
    return {
        "n_states": len(raw),
        "explained_variance": explained_variance,
        "mean_reconstruction_cosine": float(np.mean(cosine)),
        "median_reconstruction_cosine": float(np.median(cosine)),
        "mean_sae_l0": float(np.mean(l0)),
        "sae_gate_ev_at_least_0_50": bool(explained_variance >= 0.50),
        "sae_gate_cosine_at_least_0_80": bool(np.mean(cosine) >= 0.80),
        "sae_confirmatory_compatible": bool(
            explained_variance >= 0.50 and np.mean(cosine) >= 0.80
        ),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpot", "2wiki"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", default="configs/fpe5_l18.yaml")
    parser.add_argument("--n-questions", type=int, default=80)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    data_path = ROOT / "data" / f"{args.tag}_trajectory.jsonl"
    cache_path = ROOT / "outputs" / "cache" / f"{args.tag}_trajectory_states.npz"
    result_path = ROOT / "outputs" / f"{args.tag}_collection_results.json"
    assets = None
    if args.force_generate or not data_path.exists():
        assets = load_all(args.config)
        questions = source_questions(args.dataset, args.n_questions)
        rows = generate_rows(
            assets, questions, args.samples, args.temperature, args.top_p,
            args.top_k, args.max_new_tokens, args.seed, args.chat_template,
        )
        write_jsonl(rows, data_path)
    else:
        rows = read_jsonl(data_path)

    if args.force_extract or not cache_path.exists():
        if assets is None:
            assets = load_all(args.config)
        states = extract_states(assets, rows, cache_path)
    else:
        archive = np.load(cache_path, allow_pickle=True)
        states = {key: archive[key] for key in archive.files}

    if assets is None:
        assets = load_all(args.config)
    audit = reconstruction_audit(assets, states)
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    discordant = sum(
        len({row["model_correct"] for row in local}) == 2
        for local in by_question.values()
    )
    payload = {
        "experiment": args.tag,
        "dataset": args.dataset,
        "model": assets.cfg["model"]["name"],
        "chat_template": args.chat_template,
        "n_questions": len(by_question),
        "n_rows": len(rows),
        "accuracy": float(np.mean([row["model_correct"] for row in rows])),
        "discordant_final_questions": discordant,
        "sampling": {
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
            "samples": args.samples, "seed": args.seed,
        },
        "sae_reconstruction_audit": audit,
        "data_path": str(data_path),
        "cache_path": str(cache_path),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
