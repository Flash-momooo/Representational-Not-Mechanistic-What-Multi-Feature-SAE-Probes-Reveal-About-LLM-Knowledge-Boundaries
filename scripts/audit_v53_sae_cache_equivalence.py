"""Audit that V53 uses the same GemmaScope encoding as its source cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load import load_config, load_saes  # noqa: E402

SOURCE = ROOT / "outputs/cache/equal_compute_source.npz"
OUTPUT = ROOT / "outputs/poc_v53_sae_cache_equivalence.json"
BATCH_SIZE = 128
TOPK = 32


def main() -> None:
    with np.load(SOURCE, allow_pickle=True) as data:
        raw = data["raw"].astype(np.float32)
        cached = data["sae"].astype(np.float32)

    cfg = load_config(str(ROOT / "configs/fpe5_l18.yaml"))
    cfg["sae"]["layers"] = [18]
    sae = load_saes(cfg)[18]
    encoded_parts: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(raw), BATCH_SIZE):
                batch = torch.from_numpy(raw[start:start + BATCH_SIZE]).to(
                    device=cfg["model"]["device"],
                    dtype=getattr(torch, cfg["model"]["dtype"]),
                )
                encoded_parts.append(sae.encode(batch).float().cpu().numpy())
    finally:
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    encoded = np.concatenate(encoded_parts).astype(np.float32)
    flat_corr = float(np.corrcoef(cached.ravel(), encoded.ravel())[0, 1])
    cosine = np.sum(cached * encoded, axis=1) / (
        np.linalg.norm(cached, axis=1) * np.linalg.norm(encoded, axis=1) + 1e-12
    )
    cached_top = np.argpartition(cached, -TOPK, axis=1)[:, -TOPK:]
    encoded_top = np.argpartition(encoded, -TOPK, axis=1)[:, -TOPK:]
    overlap = [
        len(set(map(int, left)) & set(map(int, right))) / TOPK
        for left, right in zip(cached_top, encoded_top)
    ]
    payload = {
        "experiment": "V53 GemmaScope source-cache equivalence audit",
        "source_cache": str(SOURCE.relative_to(ROOT)),
        "rows": int(len(raw)),
        "sae": "GemmaScope layer-18 width-16k",
        "flat_activation_correlation": flat_corr,
        "per_row_cosine": {
            "mean": float(np.mean(cosine)),
            "min": float(np.min(cosine)),
            "max": float(np.max(cosine)),
        },
        "top32_active_feature_overlap": {
            "mean": float(np.mean(overlap)),
            "min": float(np.min(overlap)),
            "max": float(np.max(overlap)),
        },
        "interpretation": "This is an implementation-equivalence audit only; it does not test support stability or prediction.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
