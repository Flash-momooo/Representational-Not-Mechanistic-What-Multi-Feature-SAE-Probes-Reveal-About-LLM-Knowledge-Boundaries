"""V40: extract and evaluate machine-verifiable pre-commitment states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v37_same_question_trajectory import (
    evaluate,
    prefix_hash,
    summarize,
)
from src.extract import ResidualHook
from src.load import load_config, load_model_and_tokenizer, load_saes


LAYERS = (9, 18)
STAGES = ("T0", "T1", "T2", "C-1", "C")
OPTION_IDS = ("A", "B", "C", "D")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_generation(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def clean_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if not row["leakage_flag"] and not row["answer_mention_flag"]
    ]


def distribution_features(logits: torch.Tensor) -> np.ndarray:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    top2 = torch.topk(probs, k=2).values
    entropy = -(probs * log_probs).sum()
    return np.asarray([
        float(top2[0].detach().cpu()),
        float(entropy.detach().cpu()),
        float((top2[0] - top2[1]).detach().cpu()),
        0.0,
        0.0,
    ], dtype=np.float32)


def candidate_confidence(row: dict) -> np.ndarray:
    probs = np.asarray(list(row["option_probabilities"].values()), dtype=np.float32)
    ordered = np.sort(probs)[::-1]
    return np.asarray([
        float(ordered[0]),
        float(row["restricted_entropy"]),
        float(ordered[0] - ordered[1]),
        float(np.mean(list(row["candidate_mean_logprob"].values()))),
        float(np.std(list(row["candidate_mean_logprob"].values()))),
    ], dtype=np.float32)


def selected_candidate_confidence(row: dict) -> np.ndarray:
    probs = np.asarray(list(row["option_probabilities"].values()), dtype=np.float32)
    ordered = np.sort(probs)[::-1]
    selected_prob = float(row["option_probabilities"][row["selected_option"]])
    return np.asarray([
        selected_prob,
        float(row["restricted_entropy"]),
        float(ordered[0] - ordered[1]),
        float(ordered[0]),
        float(selected_prob >= ordered[0] - 1e-8),
    ], dtype=np.float32)


def commitment_suffix(tokenizer, item: dict) -> list[int]:
    option_text = "\n".join(f"- {item['options'][option]}" for option in OPTION_IDS)
    return [
        int(x) for x in tokenizer.encode(
            f"\nCandidates:\n{option_text}\nFINAL ANSWER:",
            add_special_tokens=False,
        )
    ]


def padded_batch(sequences: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    lengths = np.asarray([len(ids) for ids in sequences], dtype=np.int64)
    max_len = int(lengths.max())
    input_ids = torch.full(
        (len(sequences), max_len), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, ids in enumerate(sequences):
        length = len(ids)
        input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
    return input_ids, attention_mask, lengths


def encode_stage(
    assets,
    hooks: dict[int, ResidualHook],
    sequences: list[list[int]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    input_ids, attention_mask, lengths = padded_batch(
        sequences, int(assets.tokenizer.eos_token_id), assets.device
    )
    with torch.no_grad():
        outputs = assets.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    batch_index = torch.arange(len(sequences), device=input_ids.device)
    position_index = torch.tensor(lengths - 1, device=input_ids.device)
    final_logits = outputs.logits[batch_index, position_index]
    raw = {}
    sae = {}
    for layer in LAYERS:
        vectors = hooks[layer].value[batch_index, position_index]
        with torch.no_grad():
            latents = assets.saes[layer].encode(vectors)
        raw[layer] = vectors.float().cpu().numpy().astype(np.float16)
        sae[layer] = latents.float().cpu().numpy().astype(np.float16)
    confidence = np.stack([distribution_features(logit) for logit in final_logits])
    return raw, sae, confidence


def extract_states(
    assets,
    items: dict[str, dict],
    rows: list[dict],
    cache_path: Path,
) -> None:
    n_rows = len(rows)
    hidden_size = int(assets.model.config.hidden_size)
    d_sae = {layer: int(assets.saes[layer].cfg.d_sae) for layer in LAYERS}
    valid = {stage: np.zeros(n_rows, dtype=bool) for stage in STAGES}
    confidence = {stage: np.zeros((n_rows, 5), dtype=np.float32) for stage in STAGES}
    token_prefix = {stage: np.zeros((n_rows, 1024), dtype=np.float32) for stage in STAGES}
    raw = {
        stage: {layer: np.zeros((n_rows, hidden_size), dtype=np.float16) for layer in LAYERS}
        for stage in STAGES
    }
    sae = {
        stage: {layer: np.zeros((n_rows, d_sae[layer]), dtype=np.float16) for layer in LAYERS}
        for stage in STAGES
    }
    hooks = {
        layer: ResidualHook().attach(assets.model.model.layers[layer])
        for layer in LAYERS
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[row["item_id"]].append(row_index)

    try:
        for item_id, row_indices in tqdm(grouped.items(), desc="v40-extract"):
            item = items[item_id]
            prompt_ids = [
                int(x) for x in assets.tokenizer.encode(
                    item["trajectory_prompt"], add_special_tokens=True
                )
            ]
            rationale_by_row = {
                row_index: [int(x) for x in rows[row_index]["rationale_token_ids"]]
                for row_index in row_indices
            }

            # T0 is identical across all stochastic trajectories of an item.
            raw_values, sae_values, conf_values = encode_stage(assets, hooks, [prompt_ids])
            for row_index in row_indices:
                valid["T0"][row_index] = True
                confidence["T0"][row_index] = conf_values[0]
                token_prefix["T0"][row_index] = prefix_hash([], 0)
                for layer in LAYERS:
                    raw["T0"][layer][row_index] = raw_values[layer][0]
                    sae["T0"][layer][row_index] = sae_values[layer][0]

            for stage_index, stage in ((1, "T1"), (2, "T2")):
                active = [
                    row_index for row_index in row_indices
                    if len(rationale_by_row[row_index]) >= stage_index
                ]
                if not active:
                    continue
                sequences = [
                    prompt_ids + rationale_by_row[row_index][:stage_index]
                    for row_index in active
                ]
                raw_values, sae_values, conf_values = encode_stage(assets, hooks, sequences)
                for batch_index, row_index in enumerate(active):
                    valid[stage][row_index] = True
                    confidence[stage][row_index] = conf_values[batch_index]
                    token_prefix[stage][row_index] = prefix_hash(
                        rationale_by_row[row_index], stage_index
                    )
                    for layer in LAYERS:
                        raw[stage][layer][row_index] = raw_values[layer][batch_index]
                        sae[stage][layer][row_index] = sae_values[layer][batch_index]

            suffix = commitment_suffix(assets.tokenizer, item)
            sequences = [
                prompt_ids + rationale_by_row[row_index] + suffix
                for row_index in row_indices
            ]
            raw_values, sae_values, _ = encode_stage(assets, hooks, sequences)
            for batch_index, row_index in enumerate(row_indices):
                valid["C-1"][row_index] = True
                confidence["C-1"][row_index] = candidate_confidence(rows[row_index])
                token_prefix["C-1"][row_index] = prefix_hash(
                    rationale_by_row[row_index], len(rationale_by_row[row_index])
                )
                for layer in LAYERS:
                    raw["C-1"][layer][row_index] = raw_values[layer][batch_index]
                    sae["C-1"][layer][row_index] = sae_values[layer][batch_index]

            committed_sequences = []
            committed_prefixes = []
            for row_index in row_indices:
                selected_answer_ids = [
                    int(x) for x in assets.tokenizer.encode(
                        f" {item['options'][rows[row_index]['selected_option']]}",
                        add_special_tokens=False,
                    )
                ]
                committed_prefix = rationale_by_row[row_index] + selected_answer_ids
                committed_prefixes.append(committed_prefix)
                committed_sequences.append(
                    prompt_ids + rationale_by_row[row_index] + suffix + selected_answer_ids
                )
            raw_values, sae_values, _ = encode_stage(assets, hooks, committed_sequences)
            for batch_index, row_index in enumerate(row_indices):
                valid["C"][row_index] = True
                confidence["C"][row_index] = selected_candidate_confidence(rows[row_index])
                token_prefix["C"][row_index] = prefix_hash(
                    committed_prefixes[batch_index], len(committed_prefixes[batch_index])
                )
                for layer in LAYERS:
                    raw["C"][layer][row_index] = raw_values[layer][batch_index]
                    sae["C"][layer][row_index] = sae_values[layer][batch_index]
    finally:
        for hook in hooks.values():
            hook.detach()

    payload = {
        "labels": np.asarray([0 if row["model_correct"] else 1 for row in rows], dtype=np.int8),
        "question_ids": np.asarray([row["item_id"] for row in rows], dtype=object),
        "sample_indices": np.asarray([row["sample_index"] for row in rows], dtype=np.int16),
        "selected_options": np.asarray([row["selected_option"] for row in rows], dtype=object),
        "gold_options": np.asarray([row["gold_option"] for row in rows], dtype=object),
        "expected_risk": np.asarray([
            1.0 - float(row["option_probabilities"][row["gold_option"]])
            for row in rows
        ], dtype=np.float32),
        "greedy_labels": np.asarray([
            int(max(row["option_probabilities"], key=row["option_probabilities"].get) != row["gold_option"])
            for row in rows
        ], dtype=np.int8),
    }
    for stage in STAGES:
        payload[f"valid_{stage}"] = valid[stage]
        payload[f"confidence_{stage}"] = confidence[stage]
        payload[f"token_prefix_{stage}"] = token_prefix[stage]
        for layer in LAYERS:
            payload[f"raw_{stage}_L{layer}"] = raw[stage][layer]
            payload[f"sae_{stage}_L{layer}"] = sae[stage][layer]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"saved states -> {cache_path}")


def load_states(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def run_evaluation(states: dict[str, np.ndarray], output_path: Path) -> dict:
    y = states["labels"].astype(int)
    question_ids = states["question_ids"].astype(str)
    eval_rows = []
    for stage in STAGES:
        valid = states[f"valid_{stage}"].astype(bool)
        feature_sets = [
            ("confidence", None, states[f"confidence_{stage}"], "raw"),
            ("token_prefix", None, states[f"token_prefix_{stage}"], "raw"),
        ]
        for layer in LAYERS:
            raw_values = states[f"raw_{stage}_L{layer}"]
            sae_values = states[f"sae_{stage}_L{layer}"]
            feature_sets.extend([
                ("raw_residual", layer, raw_values, "raw"),
                ("sae", layer, sae_values, "sae"),
            ])
            if stage in {"T2", "C-1", "C"}:
                feature_sets.extend([
                    (
                        "raw_token_fused", layer,
                        np.hstack([raw_values, states[f"token_prefix_{stage}"]]), "raw",
                    ),
                    (
                        "sae_token_fused", layer,
                        np.hstack([sae_values, states[f"token_prefix_{stage}"]]), "sae",
                    ),
                ])
        for method, layer, features, kind in feature_sets:
            rows = evaluate(
                features.astype(np.float32), y, question_ids, valid, kind, centered=True
            )
            eval_rows.extend({
                "stage": stage,
                "layer": layer,
                "method": method,
                **row,
            } for row in rows)
    summary = summarize(eval_rows)
    summary.sort(
        key=lambda row: row["within_question_auroc_macro_mean"] or -1,
        reverse=True,
    )
    payload = {
        "experiment": "poc_v40_machine_verifiable_state_evaluation",
        "n_rows": int(len(y)),
        "n_questions": int(len(np.unique(question_ids))),
        "risk_rate": float(np.mean(y)),
        "stages": list(STAGES),
        "layers": list(LAYERS),
        "primary_subset": "no explicit format leakage and no direct candidate-answer mention",
        "summary": summary,
        "rows": eval_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved evaluation -> {output_path}")
    for row in summary[:30]:
        print(
            f"{row['stage']:>3} {row['method']:<18} L{row['layer']} "
            f"within={row['within_question_auroc_macro_mean']:.4f} "
            f"pop={row['population_auroc_mean']:.4f} n={row['n']}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--items", default="data/v40_machine_verifiable_candidates.jsonl")
    parser.add_argument("--generation", default="outputs/poc_v40_commitment_generation_results.json")
    parser.add_argument("--cache", default="outputs/cache/v40_machine_verifiable_states_clean.npz")
    parser.add_argument("--output", default="outputs/poc_v40_machine_verifiable_state_results.json")
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    items = {item["item_id"]: item for item in load_jsonl(ROOT / args.items)}
    rows = clean_rows(load_generation(ROOT / args.generation))
    if args.n_questions is not None:
        selected_ids = list(dict.fromkeys(row["item_id"] for row in rows))[: args.n_questions]
        selected_set = set(selected_ids)
        rows = [row for row in rows if row["item_id"] in selected_set]
    cache_path = ROOT / args.cache
    print(f"V40 clean rows={len(rows)} questions={len({row['item_id'] for row in rows})}")
    if args.force_extract or not cache_path.exists():
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        cfg = load_config(args.config)
        cfg["sae"]["layers"] = list(LAYERS)
        model, tokenizer = load_model_and_tokenizer(cfg)
        saes = load_saes(cfg)
        assets = SimpleNamespace(
            model=model, tokenizer=tokenizer, saes=saes,
            cfg=cfg, device=cfg["model"]["device"],
        )
        extract_states(assets, items, rows, cache_path)
    if not args.extract_only:
        run_evaluation(load_states(cache_path), ROOT / args.output)


if __name__ == "__main__":
    main()
