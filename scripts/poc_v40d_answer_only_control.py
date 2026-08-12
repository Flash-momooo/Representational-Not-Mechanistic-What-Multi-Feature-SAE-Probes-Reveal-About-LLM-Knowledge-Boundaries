"""V40d: test whether post-commitment detection is only answer identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v37_same_question_trajectory import evaluate, prefix_hash, summarize
from scripts.poc_v40_extract_and_evaluate import (
    LAYERS,
    clean_rows,
    encode_stage,
    load_generation,
    load_jsonl,
)
from src.extract import ResidualHook
from src.load import load_config, load_model_and_tokenizer, load_saes

STAGE = "A-only"
OPTION_IDS = ("A", "B", "C", "D")


def extract_answer_only(
    items: dict[str, dict],
    rows: list[dict],
    cache_path: Path,
    config_path: str,
) -> None:
    cfg = load_config(config_path)
    cfg["sae"]["layers"] = list(LAYERS)
    model, tokenizer = load_model_and_tokenizer(cfg)
    saes = load_saes(cfg)
    assets = SimpleNamespace(
        model=model, tokenizer=tokenizer, saes=saes,
        cfg=cfg, device=cfg["model"]["device"],
    )
    hooks = {
        layer: ResidualHook().attach(assets.model.model.layers[layer])
        for layer in LAYERS
    }
    raw_parts = {layer: [] for layer in LAYERS}
    sae_parts = {layer: [] for layer in LAYERS}
    confidence_parts = []
    answer_hashes = []
    metadata = []
    try:
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            sequences = []
            for row in batch:
                answer = items[row["item_id"]]["options"][row["selected_option"]]
                answer_ids = [
                    int(token_id) for token_id in tokenizer.encode(
                        f" {answer}", add_special_tokens=False
                    )
                ]
                sequences.append([
                    int(token_id) for token_id in tokenizer.encode(
                        f"FINAL ANSWER: {answer}", add_special_tokens=True
                    )
                ])
                answer_hashes.append(prefix_hash(answer_ids, len(answer_ids)))
                option_one_hot = [float(row["selected_option"] == option) for option in OPTION_IDS]
                metadata.append([
                    *option_one_hot,
                    float(len(answer_ids)),
                    float(len(answer)),
                    float(len(answer.split())),
                ])
            raw, sae, confidence = encode_stage(assets, hooks, sequences)
            confidence_parts.append(confidence)
            for layer in LAYERS:
                raw_parts[layer].append(raw[layer])
                sae_parts[layer].append(sae[layer])
    finally:
        for hook in hooks.values():
            hook.detach()

    payload = {
        "labels": np.asarray([0 if row["model_correct"] else 1 for row in rows], dtype=np.int8),
        "question_ids": np.asarray([row["item_id"] for row in rows], dtype=object),
        "confidence": np.vstack(confidence_parts),
        "answer_hash": np.stack(answer_hashes),
        "metadata": np.asarray(metadata, dtype=np.float32),
    }
    for layer in LAYERS:
        payload[f"raw_L{layer}"] = np.vstack(raw_parts[layer])
        payload[f"sae_L{layer}"] = np.vstack(sae_parts[layer])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"saved states -> {cache_path}")


def evaluate_control(cache_path: Path, output_path: Path) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    states = {key: data[key] for key in data.files}
    y = states["labels"].astype(int)
    question_ids = states["question_ids"].astype(str)
    valid = np.ones(len(y), dtype=bool)
    feature_sets = [
        ("confidence", None, states["confidence"], "raw"),
        ("answer_token_hash", None, states["answer_hash"], "raw"),
        ("option_and_length", None, states["metadata"], "raw"),
    ]
    for layer in LAYERS:
        feature_sets.extend([
            ("raw_residual", layer, states[f"raw_L{layer}"], "raw"),
            ("sae", layer, states[f"sae_L{layer}"], "sae"),
        ])
    evaluation_rows = []
    for method, layer, features, kind in feature_sets:
        rows = evaluate(
            features.astype(np.float32), y, question_ids, valid, kind, centered=True
        )
        evaluation_rows.extend({
            "stage": STAGE,
            "layer": layer,
            "method": method,
            **row,
        } for row in rows)
    summary = summarize(evaluation_rows)
    summary.sort(
        key=lambda row: row["within_question_auroc_macro_mean"] or -1,
        reverse=True,
    )
    payload = {
        "experiment": "poc_v40d_answer_only_control",
        "input": "FINAL ANSWER: <selected answer>; no question, facts, rationale, or candidate list",
        "n_rows": len(y),
        "n_questions": len(np.unique(question_ids)),
        "risk_rate": float(np.mean(y)),
        "summary": summary,
        "rows": evaluation_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved evaluation -> {output_path}")
    for row in summary:
        print(
            f"{row['method']:<18} L{row['layer']} "
            f"within={row['within_question_auroc_macro_mean']:.4f} "
            f"pop={row['population_auroc_mean']:.4f}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--items", default="data/v40b_distractor_evidence_candidates.jsonl")
    parser.add_argument("--generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--cache", default="outputs/cache/v40d_answer_only_states.npz")
    parser.add_argument("--output", default="outputs/poc_v40d_answer_only_control_results.json")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    items = {item["item_id"]: item for item in load_jsonl(ROOT / args.items)}
    rows = clean_rows(load_generation(ROOT / args.generation))
    cache_path = ROOT / args.cache
    if args.force_extract or not cache_path.exists():
        extract_answer_only(items, rows, cache_path, args.config)
    evaluate_control(cache_path, ROOT / args.output)


if __name__ == "__main__":
    main()
