"""V40f: remove the generated rationale but retain context and commitment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v37_same_question_trajectory import evaluate, summarize
from scripts.poc_v40_extract_and_evaluate import (
    LAYERS,
    clean_rows,
    commitment_suffix,
    encode_stage,
    load_generation,
    load_jsonl,
)
from src.extract import ResidualHook
from src.load import load_config, load_model_and_tokenizer, load_saes


STAGE = "C-no-rationale"


def extract_states(
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
    grouped = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[row["item_id"]].append(row_index)
    raw = {layer: np.zeros((len(rows), model.config.hidden_size), dtype=np.float16) for layer in LAYERS}
    sae = {
        layer: np.zeros((len(rows), saes[layer].cfg.d_sae), dtype=np.float16)
        for layer in LAYERS
    }
    confidence = np.zeros((len(rows), 5), dtype=np.float32)
    try:
        for item_id, row_indices in grouped.items():
            item = items[item_id]
            prompt_ids = [
                int(token_id) for token_id in tokenizer.encode(
                    item["trajectory_prompt"], add_special_tokens=True
                )
            ]
            suffix = commitment_suffix(tokenizer, item)
            sequences = []
            for row_index in row_indices:
                answer = item["options"][rows[row_index]["selected_option"]]
                answer_ids = [
                    int(token_id) for token_id in tokenizer.encode(
                        f" {answer}", add_special_tokens=False
                    )
                ]
                sequences.append(prompt_ids + suffix + answer_ids)
            raw_values, sae_values, confidence_values = encode_stage(
                assets, hooks, sequences
            )
            confidence[row_indices] = confidence_values
            for layer in LAYERS:
                raw[layer][row_indices] = raw_values[layer]
                sae[layer][row_indices] = sae_values[layer]
    finally:
        for hook in hooks.values():
            hook.detach()

    payload = {
        "labels": np.asarray([0 if row["model_correct"] else 1 for row in rows], dtype=np.int8),
        "question_ids": np.asarray([row["item_id"] for row in rows], dtype=object),
        "confidence": confidence,
    }
    for layer in LAYERS:
        payload[f"raw_L{layer}"] = raw[layer]
        payload[f"sae_L{layer}"] = sae[layer]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"saved states -> {cache_path}")


def run_evaluation(cache_path: Path, output_path: Path) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    states = {key: data[key] for key in data.files}
    labels = states["labels"].astype(int)
    question_ids = states["question_ids"].astype(str)
    valid = np.ones(len(labels), dtype=bool)
    feature_sets = [("confidence", None, states["confidence"], "raw")]
    for layer in LAYERS:
        feature_sets.extend([
            ("raw_residual", layer, states[f"raw_L{layer}"], "raw"),
            ("sae", layer, states[f"sae_L{layer}"], "sae"),
        ])
    rows = []
    for method, layer, features, kind in feature_sets:
        results = evaluate(
            features.astype(np.float32), labels, question_ids, valid, kind, centered=True
        )
        rows.extend({"stage": STAGE, "layer": layer, "method": method, **row} for row in results)
    summary = summarize(rows)
    summary.sort(
        key=lambda row: row["within_question_auroc_macro_mean"] or -1,
        reverse=True,
    )
    payload = {
        "experiment": "poc_v40f_no_rationale_control",
        "input": "facts + question + candidate list + selected answer; generated rationale removed",
        "n_rows": len(labels),
        "n_questions": len(np.unique(question_ids)),
        "summary": summary,
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved evaluation -> {output_path}")
    for row in summary:
        print(
            f"{row['method']:<14} L{row['layer']} "
            f"within={row['within_question_auroc_macro_mean']:.4f} "
            f"pop={row['population_auroc_mean']:.4f}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--items", default="data/v40b_distractor_evidence_candidates.jsonl")
    parser.add_argument("--generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--output", default="outputs/poc_v40f_no_rationale_control_results.json")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    items = {item["item_id"]: item for item in load_jsonl(ROOT / args.items)}
    rows = clean_rows(load_generation(ROOT / args.generation))
    cache_path = ROOT / args.cache
    if args.force_extract or not cache_path.exists():
        extract_states(items, rows, cache_path, args.config)
    run_evaluation(cache_path, ROOT / args.output)


if __name__ == "__main__":
    main()
