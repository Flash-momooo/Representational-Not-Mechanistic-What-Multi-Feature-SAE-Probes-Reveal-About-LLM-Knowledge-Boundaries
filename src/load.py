"""模型与 GemmaScope SAE 加载。

在 RTX 5080 16GB 上以 bf16 加载 Gemma-2-2B（约 5GB），
并通过 sae_lens 加载指定层的 GemmaScope JumpReLU SAE。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedAssets:
    model: torch.nn.Module
    tokenizer: object
    saes: Dict[int, object]   # layer -> SAE
    cfg: dict
    device: str


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(cfg: dict):
    """以 bf16 加载 Gemma-2 base 模型。"""
    name = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])
    device = cfg["model"]["device"]
    local_files_only = bool(cfg["model"].get("local_files_only", False))

    tokenizer = AutoTokenizer.from_pretrained(
        name, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",   # GemmaScope SAE 在 eager attention 下训练，保持一致
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def load_saes(cfg: dict) -> Dict[int, object]:
    """按 configs/default.yaml 中 sae.layers 加载 GemmaScope SAE。

    使用 sae_lens 的 from_pretrained API。
    """
    from sae_lens import SAE

    release = cfg["sae"]["release"]
    width = cfg["sae"]["width"]
    layers: List[int] = cfg["sae"]["layers"]
    device = cfg["model"]["device"]
    dtype = getattr(torch, cfg["model"]["dtype"])

    saes: Dict[int, object] = {}
    for layer in layers:
        # GemmaScope canonical naming: layer_{L}/width_{W}/canonical
        sae_id = f"layer_{layer}/width_{width}/canonical"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        sae = sae.to(dtype=dtype)
        sae.eval()
        saes[layer] = sae
        # 兼容新旧 sae_lens API：hook_name / d_sae 可能在 cfg 或 cfg.metadata 下
        d_sae = getattr(sae.cfg, "d_sae", None) \
            or getattr(getattr(sae.cfg, "metadata", None), "d_sae", "?")
        hook_name = getattr(sae.cfg, "hook_name", None) \
            or getattr(getattr(sae.cfg, "metadata", None), "hook_name", "?")
        print(f"[load] SAE loaded: layer={layer}, d_sae={d_sae}, hook={hook_name}")
    return saes


def load_all(config_path: str = "configs/default.yaml") -> LoadedAssets:
    cfg = load_config(config_path)
    print(f"[load] config: {cfg['model']['name']} | layers {cfg['sae']['layers']} | width {cfg['sae']['width']}")
    model, tokenizer = load_model_and_tokenizer(cfg)
    print(f"[load] model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    saes = load_saes(cfg)
    return LoadedAssets(
        model=model,
        tokenizer=tokenizer,
        saes=saes,
        cfg=cfg,
        device=cfg["model"]["device"],
    )


if __name__ == "__main__":
    assets = load_all()
    print("[load] OK")
    print(f"  device: {assets.device}")
    print(f"  layers: {list(assets.saes.keys())}")
