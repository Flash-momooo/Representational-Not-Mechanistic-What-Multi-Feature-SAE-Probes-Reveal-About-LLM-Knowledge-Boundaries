"""FPE19 measured latency of dense and SAE sparse hallucination readouts."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load import load_all


CONFIG = ROOT / "configs/fpe5_l18.yaml"
CACHE = ROOT / "outputs/cache/fpe14_gemma_trivia_confirmatory_trajectory_states.npz"
DATA = ROOT / "data/fpe14_gemma_trivia_confirmatory_trajectory.jsonl"
BATCH_SIZES = (1, 16, 64)
WARMUP = 20
REPEATS = 100
K = 8


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cuda_benchmark(operation, warmup: int = WARMUP, repeats: int = REPEATS) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return {
        "median_ms": float(statistics.median(samples)),
        "p05_ms": float(np.quantile(samples, 0.05)),
        "p95_ms": float(np.quantile(samples, 0.95)),
        "repeats": repeats,
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("FPE19 requires CUDA")
    assets = load_all(CONFIG)
    model, tokenizer, sae = assets.model, assets.tokenizer, assets.saes[18]
    archive = np.load(CACHE, allow_pickle=True)
    raw = torch.from_numpy(archive["raw_T0"].astype(np.float32)).to(
        device="cuda", dtype=torch.bfloat16
    )
    d_model = int(raw.shape[1])
    d_sae = int(sae.cfg.d_sae)
    dense_weight = torch.randn(d_model, device="cuda", dtype=torch.bfloat16)
    sparse_indices = torch.arange(K, device="cuda")
    sparse_weight = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    rows = read_jsonl(DATA)
    unique_prompts = {}
    for row in rows:
        unique_prompts.setdefault(str(row["question_id"]), row["prompt"])
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in unique_prompts.values()
    ]
    median_question = sorted(unique_prompts)[len(unique_prompts) // 2]
    model_input = tokenizer(
        unique_prompts[median_question], return_tensors="pt"
    )["input_ids"].to(model.device)

    payload = {
        "experiment": "FPE19 measured monitor cost",
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "d_model": d_model,
        "d_sae": d_sae,
        "k": K,
        "input_dimension_compression": d_model / K,
        "prompt_tokens": {
            "median": float(np.median(prompt_lengths)),
            "p05": float(np.quantile(prompt_lengths, 0.05)),
            "p95": float(np.quantile(prompt_lengths, 0.95)),
            "benchmarked": int(model_input.shape[1]),
        },
        "batch_results": [],
    }
    for batch_size in BATCH_SIZES:
        vectors = raw[:batch_size]
        with torch.inference_mode():
            latents = sae.encode(vectors)
        dense = cuda_benchmark(lambda: vectors @ dense_weight)
        sparse_only = cuda_benchmark(
            lambda: latents[:, sparse_indices] @ sparse_weight
        )
        sae_end_to_end = cuda_benchmark(
            lambda: sae.encode(vectors)[:, sparse_indices] @ sparse_weight
        )
        payload["batch_results"].append({
            "batch_size": batch_size,
            "dense_linear_readout": dense,
            "sparse_k8_readout_given_latents": sparse_only,
            "sae_encode_plus_sparse_k8": sae_end_to_end,
            "dense_us_per_sample": 1000.0 * dense["median_ms"] / batch_size,
            "sparse_given_latents_us_per_sample": 1000.0 * sparse_only["median_ms"] / batch_size,
            "sae_end_to_end_us_per_sample": 1000.0 * sae_end_to_end["median_ms"] / batch_size,
        })
    forward = cuda_benchmark(
        lambda: model(input_ids=model_input, use_cache=False),
        warmup=10,
        repeats=50,
    )
    payload["single_prompt_transformer_forward"] = forward
    with torch.inference_mode():
        prefill = model(input_ids=model_input, use_cache=True)
        next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        past_key_values = prefill.past_key_values
    decode = cuda_benchmark(
        lambda: model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        ),
        warmup=10,
        repeats=50,
    )
    payload["single_token_cached_decode"] = decode
    batch_one = payload["batch_results"][0]
    payload["batch_one_sae_over_forward_fraction"] = (
        batch_one["sae_encode_plus_sparse_k8"]["median_ms"] / forward["median_ms"]
    )
    payload["batch_one_sae_over_cached_decode_fraction"] = (
        batch_one["sae_encode_plus_sparse_k8"]["median_ms"] / decode["median_ms"]
    )
    payload["interpretation"] = (
        "K=8 is a readout-dimension reduction. Actual overhead includes the full SAE encoder; "
        "report measured latency rather than treating dimensional compression as speedup. "
        "The benchmark excludes Python hook overhead and assumes the residual is already GPU-resident."
    )
    output = ROOT / "outputs/poc_fpe19_monitor_cost_results.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
