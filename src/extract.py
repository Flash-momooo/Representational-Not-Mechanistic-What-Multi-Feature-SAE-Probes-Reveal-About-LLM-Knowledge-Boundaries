"""激活提取与 SAE 编码。

Ferrando 设定：在 entity 的**最后一个 token** 位置读取 residual stream 激活，
再用 GemmaScope SAE 编码得到稀疏特征向量。

实现方式：
1. 用 tokenizer 分别编码 prompt 与 entity，定位 entity 最后一个 token 在完整 prompt 中的索引。
2. forward 时用 hook 抓取指定层的 residual stream（hook_name 来自 SAE config）。
3. 调用 sae.encode 得到稀疏特征。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data.entity_qa import EntityExample
from .load import LoadedAssets


# ----------------------------------------------------------------------
# Hook：抓取指定层 residual stream 的输出
# ----------------------------------------------------------------------
class ResidualHook:
    """注册到 model.model.layers[L] 的 forward hook，缓存其输出。"""
    def __init__(self):
        self.value: torch.Tensor | None = None
        self.handle = None

    def __call__(self, module, inputs, output):
        # Gemma-2 decoder layer 输出: (hidden_states, ...)
        if isinstance(output, tuple):
            self.value = output[0]
        else:
            self.value = output

    def attach(self, layer_module: torch.nn.Module):
        self.handle = layer_module.register_forward_hook(self)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ----------------------------------------------------------------------
# Entity token 定位
# ----------------------------------------------------------------------
def find_entity_last_token_index(tokenizer, prompt: str, entity: str) -> int:
    """返回 entity 最后一个 token 在 tokenize(prompt) 后的索引（0-based）。

    策略：在 prompt 中找到 entity 的最后一次出现的字符位置 end_char，
    使用 fast tokenizer 的 offset_mapping 反查 token 索引。
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    # 找 entity 最后一次出现
    start_char = prompt.rfind(entity)
    if start_char < 0:
        # 兜底：返回最后一个 token
        return len(offsets) - 1
    end_char = start_char + len(entity)  # exclusive
    # 寻找覆盖到 end_char-1 的 token
    target = end_char - 1
    last_idx = len(offsets) - 1
    for i, (s, e) in enumerate(offsets):
        if s <= target < e:
            return i
    return last_idx


# ----------------------------------------------------------------------
# 主提取函数
# ----------------------------------------------------------------------
@dataclass
class FeatureBatch:
    """一层激活：SAE 稀疏特征 + 可选的原始 residual。"""
    layer: int
    features: np.ndarray         # (N, d_sae)  SAE encode 后
    labels: np.ndarray           # (N,)
    entity_types: List[str]
    entities: List[str]
    raw_residuals: np.ndarray | None = None    # (N, d_model) 原始 residual stream（baseline 用）


def extract_features(
    assets: LoadedAssets,
    examples: List[EntityExample],
    layers: List[int] | None = None,
    include_raw: bool = False,
) -> Dict[int, FeatureBatch]:
    """对每条 example，在每个目标层提取 entity-last-token 的 SAE 稀疏特征。

    Args:
        include_raw: True 时同时保存原始 residual stream 激活（用于 baseline 对比）

    返回 {layer: FeatureBatch}。
    """
    model = assets.model
    tokenizer = assets.tokenizer
    saes = assets.saes
    device = assets.device

    target_layers = layers if layers is not None else list(saes.keys())
    # 注册 hooks
    hooks: Dict[int, ResidualHook] = {}
    for L in target_layers:
        # Gemma-2 decoder layers 在 model.model.layers
        layer_module = model.model.layers[L]
        hooks[L] = ResidualHook().attach(layer_module)

    # 缓冲区
    feats: Dict[int, List[np.ndarray]] = {L: [] for L in target_layers}
    raws: Dict[int, List[np.ndarray]] = {L: [] for L in target_layers} if include_raw else {}
    labels: List[int] = []
    types: List[str] = []
    ents: List[str] = []

    try:
        for ex in tqdm(examples, desc="extract"):
            tok_idx = find_entity_last_token_index(tokenizer, ex.prompt, ex.entity)
            inputs = tokenizer(ex.prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)
            for L in target_layers:
                resid = hooks[L].value           # (1, T, d_model)
                vec = resid[0, tok_idx, :]        # (d_model,)
                if include_raw:
                    raws[L].append(vec.float().cpu().numpy())
                # SAE encode
                with torch.no_grad():
                    z = saes[L].encode(vec.unsqueeze(0)).squeeze(0)   # (d_sae,)
                feats[L].append(z.float().cpu().numpy())
            labels.append(ex.label)
            types.append(ex.entity_type)
            ents.append(ex.entity)
    finally:
        for h in hooks.values():
            h.detach()

    out = {}
    labels_arr = np.array(labels, dtype=np.int64)
    for L in target_layers:
        F = np.stack(feats[L], axis=0)
        R = np.stack(raws[L], axis=0) if include_raw else None
        out[L] = FeatureBatch(layer=L, features=F, labels=labels_arr,
                              entity_types=types, entities=ents,
                              raw_residuals=R)
    return out


if __name__ == "__main__":
    from .data.entity_qa import get_or_build
    from .load import load_all

    assets = load_all()
    items = get_or_build()[:8]    # 烟雾测试
    out = extract_features(assets, items)
    for L, fb in out.items():
        print(f"layer {L}: features shape={fb.features.shape}, "
              f"L0_avg={(fb.features > 0).sum(1).mean():.1f}")
