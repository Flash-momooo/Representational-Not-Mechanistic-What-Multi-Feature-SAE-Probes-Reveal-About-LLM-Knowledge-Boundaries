"""Collect model-family-generic FPE10 trajectories and residual states."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe5_confirmatory_collect import distribution_features
from scripts.poc_v37_same_question_trajectory import (
    clean_generated_ids,
    prefix_hash,
    score_answer,
)
from src.extract import ResidualHook
from src.load import load_all


STAGES = (0, 1, 2, 3)
DEPTH_FRACTION = 18 / 26
PROMPT_TEMPLATE = (
    "Answer the question. Return only the shortest answer with no explanation.\n"
    "Question: {question}\nAnswer:"
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def score_aliases(response: str, aliases: list[str]) -> tuple[bool, bool, float, str]:
    scored = [(score_answer(response, alias), alias) for alias in aliases]
    (correct, exact, f1), matched = max(
        scored, key=lambda item: (item[0][0], item[0][1], item[0][2])
    )
    return correct, exact, f1, matched


def model_layers(model):
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise AttributeError("Unsupported model: decoder layer list not found")


def format_prompt(tokenizer, question: str, chat_template: bool) -> str:
    content = (
        question
        if question.lstrip().lower().startswith("answer using only")
        else PROMPT_TEMPLATE.format(question=question)
    )
    if not chat_template:
        return content
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_name: str, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
        "attn_implementation": "eager",
        "local_files_only": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    questions: list[dict],
    samples: int,
    seed: int,
    chat_template: bool,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 16,
) -> tuple[list[dict], dict]:
    rows = []
    eos_id = int(tokenizer.eos_token_id)
    stop_ids = {eos_id}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated_tokens = 0
    for question_index, question in enumerate(tqdm(questions, desc="fpe10-generate")):
        prompt = format_prompt(tokenizer, question["prompt"], chat_template)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        torch.manual_seed(seed + question_index)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=samples,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        aliases = question.get("gold_answers") or [question["gold_answer"]]
        for sample_index, sequence in enumerate(generated):
            token_ids = clean_generated_ids(
                sequence[prompt_length:].detach().cpu().tolist(), stop_ids
            )
            generated_tokens += len(token_ids)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            nonempty = [line.strip() for line in decoded.splitlines() if line.strip()]
            response = nonempty[0] if nonempty else decoded.strip()
            response = re.sub(r"^answer\s*:\s*", "", response, flags=re.IGNORECASE).strip()
            correct, exact, f1, matched = score_aliases(response, aliases)
            rows.append({
                **question,
                "prompt": prompt,
                "sample_index": sample_index,
                "model_answer": response,
                "generated_token_ids": token_ids,
                "model_correct": correct,
                "normalized_exact_match": exact,
                "token_f1": f1,
                "matched_gold_answer": matched,
            })
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return rows, {
        "wall_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / max(elapsed, 1e-9),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }


def extract(
    model, tokenizer, rows: list[dict], layer_index: int, sae=None
) -> tuple[dict, dict]:
    layers = model_layers(model)
    hook = ResidualHook().attach(layers[layer_index])
    n_rows = len(rows)
    hidden_size = int(model.config.hidden_size)
    valid = {stage: np.zeros(n_rows, dtype=bool) for stage in STAGES}
    confidence = {stage: np.zeros((n_rows, 5), dtype=np.float32) for stage in STAGES}
    token_prefix = {stage: np.zeros((n_rows, 1024), dtype=np.float32) for stage in STAGES}
    raw = {stage: np.zeros((n_rows, hidden_size), dtype=np.float16) for stage in STAGES}
    latent = None
    if sae is not None:
        d_sae = int(sae.cfg.d_sae)
        latent = {
            stage: np.zeros((n_rows, d_sae), dtype=np.float16)
            for stage in STAGES
        }
    generated = [list(map(int, row["generated_token_ids"])) for row in rows]
    selected_log_probs = [[] for _ in rows]
    grouped = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[row["question_id"]].append(row_index)
        for stage in STAGES:
            valid[stage][row_index] = stage == 0 or len(generated[row_index]) >= stage
            token_prefix[stage][row_index] = prefix_hash(generated[row_index], stage)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    states_processed = 0
    try:
        for _, row_indices in tqdm(grouped.items(), desc="fpe10-extract"):
            prompt_ids = tokenizer(
                rows[row_indices[0]]["prompt"], return_tensors="pt"
            )["input_ids"].to(model.device)
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
                        device=model.device,
                    )
                    input_ids = torch.cat(
                        [prompt_ids.expand(len(active), -1), prefixes], dim=1
                    )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                    )
                logits = output.logits[:, -1]
                vectors = hook.value[:, -1]
                raw_values = vectors.float().cpu().numpy().astype(np.float16)
                latent_values = None
                if sae is not None:
                    with torch.inference_mode():
                        latent_values = sae.encode(vectors).float().cpu().numpy().astype(np.float16)
                states_processed += len(active)
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
                    if latent is not None:
                        latent[stage][row_index] = latent_values[source_index]
                    if len(generated[row_index]) > stage:
                        next_token = generated[row_index][stage]
                        log_prob = torch.log_softmax(logits[source_index].float(), -1)[next_token]
                        selected_log_probs[row_index].append(float(log_prob.detach().cpu()))
    finally:
        hook.detach()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "labels": np.asarray([0 if row["model_correct"] else 1 for row in rows], dtype=np.int8),
        "question_ids": np.asarray([row["question_id"] for row in rows], dtype=object),
        "pair_ids": np.asarray([row["pair_id"] for row in rows], dtype=object),
        "difficulty": np.asarray([row["difficulty"] for row in rows], dtype=object),
        "answer_lengths": np.asarray([len(row["generated_token_ids"]) for row in rows], dtype=np.int16),
        "layer_index": np.asarray(layer_index, dtype=np.int16),
        "hidden_size": np.asarray(hidden_size, dtype=np.int16),
    }
    for stage in STAGES:
        payload[f"valid_T{stage}"] = valid[stage]
        payload[f"confidence_T{stage}"] = confidence[stage]
        payload[f"token_prefix_T{stage}"] = token_prefix[stage]
        payload[f"raw_T{stage}"] = raw[stage]
        if latent is not None:
            payload[f"sae_T{stage}"] = latent[stage]
    timing = {
        "wall_seconds": elapsed,
        "states_processed": states_processed,
        "states_per_second": states_processed / max(elapsed, 1e-9),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
    return payload, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gemma-sae-config")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    data_path = ROOT / "data" / f"{args.tag}_trajectory.jsonl"
    cache_path = ROOT / "outputs" / "cache" / f"{args.tag}_trajectory_states.npz"
    result_path = ROOT / "outputs" / f"{args.tag}_collection_results.json"
    questions = read_jsonl(ROOT / args.questions)
    if args.offset:
        questions = questions[args.offset:]
    if args.limit is not None:
        questions = questions[:args.limit]
    model = tokenizer = sae = None
    generation_timing = None
    if args.force_generate or not data_path.exists():
        if args.gemma_sae_config:
            assets = load_all(args.gemma_sae_config)
            model, tokenizer = assets.model, assets.tokenizer
            sae = assets.saes[18]
        else:
            model, tokenizer = load_model(args.model, args.load_in_4bit)
        rows, generation_timing = generate(
            model,
            tokenizer,
            questions,
            args.samples,
            args.seed,
            args.chat_template,
            args.temperature,
            args.top_p,
            args.top_k,
            args.max_new_tokens,
        )
        write_jsonl(rows, data_path)
    else:
        rows = read_jsonl(data_path)
    if args.generate_only:
        labels = np.asarray(
            [0 if row["model_correct"] else 1 for row in rows], dtype=np.int8
        )
        by_question = defaultdict(list)
        for row in rows:
            by_question[row["question_id"]].append(bool(row["model_correct"]))
        payload = {
            "experiment": args.tag,
            "model": args.model,
            "questions": args.questions,
            "n_questions": len(by_question),
            "n_rows": len(rows),
            "accuracy": float(np.mean(labels == 0)),
            "n_correct": int(np.sum(labels == 0)),
            "n_error": int(np.sum(labels == 1)),
            "discordant_questions": int(
                sum(len(set(values)) == 2 for values in by_question.values())
            ),
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "samples": args.samples,
                "seed": args.seed,
            },
            "generation_only": True,
            "load_in_4bit": args.load_in_4bit,
            "generation_timing": generation_timing,
            "data_path": str(data_path.relative_to(ROOT)),
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    if args.force_extract or not cache_path.exists():
        if model is None:
            if args.gemma_sae_config:
                assets = load_all(args.gemma_sae_config)
                model, tokenizer = assets.model, assets.tokenizer
                sae = assets.saes[18]
            else:
                model, tokenizer = load_model(args.model, args.load_in_4bit)
        layers = model_layers(model)
        layer_index = int(round(DEPTH_FRACTION * len(layers)))
        layer_index = min(max(layer_index, 0), len(layers) - 1)
        if sae is not None and layer_index != 18:
            raise ValueError("GemmaScope confirmation requires frozen layer 18")
        states, extraction_timing = extract(
            model, tokenizer, rows, layer_index, sae=sae
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **states)
    else:
        archive = np.load(cache_path, allow_pickle=True)
        states = {key: archive[key] for key in archive.files}
        layer_index = int(states["layer_index"])
        extraction_timing = None
    labels = states["labels"].astype(np.int8)
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(bool(row["model_correct"]))
    payload = {
        "experiment": args.tag,
        "model": args.model,
        "questions": args.questions,
        "n_questions": len(by_question),
        "n_rows": len(rows),
        "accuracy": float(np.mean(labels == 0)),
        "n_correct": int(np.sum(labels == 0)),
        "n_error": int(np.sum(labels == 1)),
        "discordant_questions": int(sum(len(set(values)) == 2 for values in by_question.values())),
        "layer_index": layer_index,
        "normalized_depth": DEPTH_FRACTION,
        "sae_encoded": bool(args.gemma_sae_config),
        "load_in_4bit": args.load_in_4bit,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "samples": args.samples,
            "seed": args.seed,
        },
        "generation_timing": generation_timing,
        "extraction_timing": extraction_timing,
        "data_path": str(data_path.relative_to(ROOT)),
        "cache_path": str(cache_path.relative_to(ROOT)),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
