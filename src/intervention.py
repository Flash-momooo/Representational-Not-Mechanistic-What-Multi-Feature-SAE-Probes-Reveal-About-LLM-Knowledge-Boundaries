"""V7：SAE 特征因果干预（Feature Ablation / Amplification）。

核心：在 L 层 residual stream forward hook 中减去目标特征的贡献：
    r_new = r - scale * (sae.encode(r)[feature_idx] * sae.W_dec[feature_idx])

scale 控制干预强度：
    scale = 1.0  → 完全 ablation（移除该特征对 residual 的贡献）
    scale = 0.0  → 无干预
    scale < 0    → amplification（将特征贡献放大 |scale|+1 倍）

提供两种 causal metric：
    A. Generation-based（V7）：ablate 后看 model.generate 答对率 — 受 greedy decoding 容忍度限制
    B. Logit-based（V7b）：ablate 后看 log P(gold_answer | prompt) 变化 — mech interp 标准
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data.entity_qa import EntityExample
from .data.self_knowledge import FEW_SHOT_TEMPLATE, is_answer_correct


# ----------------------------------------------------------------------
# Hook：在指定层 residual stream 中减去 target feature 的贡献
# ----------------------------------------------------------------------
class FeatureAblationHook:
    """前向 hook：单特征 ablation
       r_new = r - scale * (act[idx] * W_dec[idx])

    scale=1.0 ablation；scale=0.0 no-op；scale=-K amplification（贡献变 K+1 倍）。
    """
    def __init__(self, sae, feature_idx: int, scale: float = 1.0):
        self.sae = sae
        self.feature_idx = feature_idx
        self.scale = scale
        self.handle = None
        self._fire_count = 0

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            resid = output[0]
            rest = output[1:]
        else:
            resid = output
            rest = ()

        with torch.no_grad():
            feats = self.sae.encode(resid)                                 # (B, T, d_sae)
            target_act = feats[..., self.feature_idx]                     # (B, T)
            w_dec = self.sae.W_dec[self.feature_idx]                       # (d_model,)
            contribution = target_act.unsqueeze(-1) * w_dec                # (B, T, d_model)
            resid_new = resid - self.scale * contribution

        self._fire_count += 1

        if rest:
            return (resid_new,) + rest
        return resid_new

    def attach(self, layer_module: torch.nn.Module):
        self.handle = layer_module.register_forward_hook(self)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MultiFeatureAblationHook:
    """前向 hook：联合多特征 ablation
       r_new = r - scale * Σ_{idx in S} (act[idx] * W_dec[idx])

    用一次 encode 后向量化处理多个特征 → 比逐个 hook 快 N 倍。
    """
    def __init__(self, sae, feature_indices: List[int], scale: float = 1.0):
        self.sae = sae
        self.feature_indices = torch.tensor(list(feature_indices), dtype=torch.long)
        self.scale = scale
        self.handle = None
        self._fire_count = 0

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            resid = output[0]
            rest = output[1:]
        else:
            resid = output
            rest = ()

        with torch.no_grad():
            feats = self.sae.encode(resid)                                 # (B, T, d_sae)
            idx = self.feature_indices.to(resid.device)
            target_acts = feats[..., idx]                                  # (B, T, K)
            w_decs = self.sae.W_dec[idx]                                   # (K, d_model)
            # contribution = sum over K of act[k] * w_dec[k]
            #              = (B, T, K) @ (K, d_model) = (B, T, d_model)
            contribution = target_acts @ w_decs
            resid_new = resid - self.scale * contribution

        self._fire_count += 1

        if rest:
            return (resid_new,) + rest
        return resid_new

    def attach(self, layer_module: torch.nn.Module):
        self.handle = layer_module.register_forward_hook(self)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ----------------------------------------------------------------------
# 单条生成（带可选 ablation hook）
# ----------------------------------------------------------------------
def generate_with_optional_ablation(
    model,
    tokenizer,
    question: str,
    sae=None,
    layer_idx: int | None = None,
    feature_idx: int | None = None,
    scale: float = 0.0,
    max_new_tokens: int = 20,
) -> str:
    """若 sae+feature_idx 都给定且 scale != 0，则注册 ablation hook。否则普通生成。"""
    hook = None
    if sae is not None and feature_idx is not None and scale != 0.0:
        hook = FeatureAblationHook(sae, feature_idx, scale=scale)
        hook.attach(model.model.layers[layer_idx])
    try:
        prompt = FEW_SHOT_TEMPLATE.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop in ["\nQ:", "\n\n", "\n"]:
            if stop in text:
                text = text.split(stop)[0]
                break
        return text.strip()
    finally:
        if hook is not None:
            hook.detach()


# ----------------------------------------------------------------------
# V7b：Logit-based causal metric（mech interp 标准）
# ----------------------------------------------------------------------
def compute_gold_logprob(
    model,
    tokenizer,
    prompt: str,
    gold_answer: str,
    sae=None,
    layer: int | None = None,
    feature_idx: int | None = None,
    scale: float = 0.0,
) -> Tuple[float, float, int]:
    """计算 log P(gold_answer | prompt)（可带 ablation hook）。

    返回 (total_logprob, avg_logprob_per_token, n_answer_tokens)。

    实现：把 "<prompt> <gold_answer>" 一起喂入，取 answer 部分每个 token 的 log P。
    """
    # 拼接 prompt + 答案，gold_answer 前加空格保证与 BPE 一致
    full = prompt + " " + gold_answer
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True).to(model.device)
    prompt_len = prompt_ids["input_ids"].shape[1]
    answer_len = full_ids["input_ids"].shape[1] - prompt_len
    if answer_len <= 0:
        return 0.0, 0.0, 0

    hook = None
    if sae is not None and feature_idx is not None and scale != 0.0:
        hook = FeatureAblationHook(sae, feature_idx, scale=scale)
        hook.attach(model.model.layers[layer])
    try:
        with torch.no_grad():
            out = model(**full_ids, use_cache=False)
        logits = out.logits[0]                                 # (T, V)
        log_probs = torch.log_softmax(logits.float(), dim=-1)  # 升 fp32 避免 bf16 精度
        # logits[i] 预测 token[i+1]，所以 answer 第 j 个 token 由 logits[prompt_len-1+j] 预测
        answer_tokens = full_ids["input_ids"][0, prompt_len:]
        log_p = log_probs[prompt_len - 1: prompt_len - 1 + answer_len].gather(
            1, answer_tokens.unsqueeze(-1)).squeeze(-1)
        total = log_p.sum().item()
        avg = total / answer_len
        return total, avg, answer_len
    finally:
        if hook is not None:
            hook.detach()


@dataclass
class LogitInterventionResult:
    feature_spec: str
    layer: int
    feature_idx: int
    scale: float
    n_known: int
    n_unknown: int
    # 每条样本的 log-prob 差异：abl - base（负值 = ablation 让 gold 答案更不可能）
    delta_logp_known: List[float]
    delta_logp_unknown: List[float]
    # 统计摘要
    mean_delta_known: float
    mean_delta_unknown: float
    std_delta_known: float
    std_delta_unknown: float


def run_logit_intervention(
    assets,
    items: List[EntityExample],
    layer: int,
    feature_idx: int,
    scale: float = 1.0,
    n_known: int = 50,
    n_unknown: int = 50,
    show_progress: bool = True,
) -> LogitInterventionResult:
    """对 known/unknown 样本各 N 条，比较 baseline vs ablated 的 log P(gold) 差异。

    Note: 使用 RAW PROMPT（与 probe 训练一致），不使用 few-shot 模板。
    """
    sae = assets.saes[layer]
    model = assets.model
    tokenizer = assets.tokenizer

    known = [x for x in items if x.model_correct is True][:n_known]
    unknown = [x for x in items if x.model_correct is False][:n_unknown]

    feature_spec = f"L{layer} #{feature_idx}"
    print(f"  intervening on {feature_spec} (scale={scale})  | known={len(known)} unknown={len(unknown)}")

    def _delta_for(group_items, group_name):
        deltas = []
        for it in tqdm(group_items, desc=f"  {group_name:7s}", disable=not show_progress):
            gold = (it.gold_answers or [None])[0]
            if not gold:
                continue
            # baseline：scale=0
            logp_base, _, _ = compute_gold_logprob(
                model, tokenizer, it.prompt, gold,
                sae=None, layer=None, feature_idx=None, scale=0.0,
            )
            # ablated：scale=scale
            logp_abl, _, _ = compute_gold_logprob(
                model, tokenizer, it.prompt, gold,
                sae=sae, layer=layer, feature_idx=feature_idx, scale=scale,
            )
            deltas.append(logp_abl - logp_base)
        return deltas

    delta_k = _delta_for(known, "known")
    delta_u = _delta_for(unknown, "unknown")

    return LogitInterventionResult(
        feature_spec=feature_spec,
        layer=layer,
        feature_idx=feature_idx,
        scale=scale,
        n_known=len(delta_k),
        n_unknown=len(delta_u),
        delta_logp_known=delta_k,
        delta_logp_unknown=delta_u,
        mean_delta_known=float(np.mean(delta_k)) if delta_k else 0.0,
        mean_delta_unknown=float(np.mean(delta_u)) if delta_u else 0.0,
        std_delta_known=float(np.std(delta_k)) if delta_k else 0.0,
        std_delta_unknown=float(np.std(delta_u)) if delta_u else 0.0,
    )


# ----------------------------------------------------------------------
# V7c：多特征联合 ablation 的 logit-based metric
# ----------------------------------------------------------------------
def compute_gold_logprob_multi(
    model,
    tokenizer,
    prompt: str,
    gold_answer: str,
    sae,
    layer: int,
    feature_indices: List[int],
    scale: float = 1.0,
) -> Tuple[float, float, int]:
    """与 compute_gold_logprob 同接口，但 ablate 多个特征。"""
    full = prompt + " " + gold_answer
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True).to(model.device)
    prompt_len = prompt_ids["input_ids"].shape[1]
    answer_len = full_ids["input_ids"].shape[1] - prompt_len
    if answer_len <= 0:
        return 0.0, 0.0, 0

    hook = None
    if feature_indices and scale != 0.0:
        hook = MultiFeatureAblationHook(sae, feature_indices, scale=scale)
        hook.attach(model.model.layers[layer])
    try:
        with torch.no_grad():
            out = model(**full_ids, use_cache=False)
        logits = out.logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        answer_tokens = full_ids["input_ids"][0, prompt_len:]
        log_p = log_probs[prompt_len - 1: prompt_len - 1 + answer_len].gather(
            1, answer_tokens.unsqueeze(-1)).squeeze(-1)
        total = log_p.sum().item()
        avg = total / answer_len
        return total, avg, answer_len
    finally:
        if hook is not None:
            hook.detach()


@dataclass
class MultiAblationResult:
    layer: int
    K: int
    feature_indices: List[int]
    scale: float
    n_known: int
    n_unknown: int
    mean_delta_known: float
    std_delta_known: float
    mean_delta_unknown: float
    std_delta_unknown: float
    delta_logp_known: List[float]
    delta_logp_unknown: List[float]


def run_multi_feature_intervention(
    assets,
    items: List[EntityExample],
    layer: int,
    feature_indices: List[int],
    scale: float = 1.0,
    n_known: int = 50,
    n_unknown: int = 50,
    show_progress: bool = True,
) -> MultiAblationResult:
    """对 known/unknown 各 N 条样本，联合 ablate feature_indices 中的所有特征。"""
    sae = assets.saes[layer]
    model = assets.model
    tokenizer = assets.tokenizer

    known = [x for x in items if x.model_correct is True][:n_known]
    unknown = [x for x in items if x.model_correct is False][:n_unknown]
    K = len(feature_indices)

    def _delta_for(group_items, group_name):
        deltas = []
        for it in tqdm(group_items, desc=f"  K={K:3d} {group_name:7s}", disable=not show_progress):
            gold = (it.gold_answers or [None])[0]
            if not gold:
                continue
            logp_base, _, _ = compute_gold_logprob(
                model, tokenizer, it.prompt, gold,
                sae=None, layer=None, feature_idx=None, scale=0.0,
            )
            logp_abl, _, _ = compute_gold_logprob_multi(
                model, tokenizer, it.prompt, gold,
                sae=sae, layer=layer, feature_indices=feature_indices, scale=scale,
            )
            deltas.append(logp_abl - logp_base)
        return deltas

    delta_k = _delta_for(known, "known")
    delta_u = _delta_for(unknown, "unknown")

    return MultiAblationResult(
        layer=layer,
        K=K,
        feature_indices=list(feature_indices),
        scale=scale,
        n_known=len(delta_k),
        n_unknown=len(delta_u),
        mean_delta_known=float(np.mean(delta_k)) if delta_k else 0.0,
        std_delta_known=float(np.std(delta_k)) if delta_k else 0.0,
        mean_delta_unknown=float(np.mean(delta_u)) if delta_u else 0.0,
        std_delta_unknown=float(np.std(delta_u)) if delta_u else 0.0,
        delta_logp_known=delta_k,
        delta_logp_unknown=delta_u,
    )


# ----------------------------------------------------------------------
# V7（旧版）：Generation-based causal metric
# ----------------------------------------------------------------------
@dataclass
class InterventionResult:
    feature_spec: str                  # e.g. "L18 #2740"
    layer: int
    feature_idx: int
    scale: float
    n_known: int
    n_unknown: int
    baseline_known_acc: float          # 已知样本 baseline 准确率（应接近 1.0）
    ablated_known_acc: float           # 已知样本 ablation 后准确率
    delta_known: float                 # ablated - baseline；负值越大干预效果越显著
    baseline_unknown_acc: float        # 未知样本 baseline 准确率（应接近 0）
    ablated_unknown_acc: float         # 未知样本 ablation 后准确率
    delta_unknown: float
    per_example_changes: List[dict]    # 详细记录每条样本的前后变化


def run_intervention_experiment(
    assets,
    items: List[EntityExample],
    layer: int,
    feature_idx: int,
    scale: float = 1.0,
    n_known: int = 50,
    n_unknown: int = 50,
    show_progress: bool = True,
) -> InterventionResult:
    """对一组 V4 self-labeled 样本运行 baseline + ablation 对照。

    items 中每条须有 model_correct 字段（V4 标注好的）。
    """
    sae = assets.saes[layer]
    model = assets.model
    tokenizer = assets.tokenizer

    # 抽样：known 和 unknown 各取前 n_per_class 条
    known = [x for x in items if x.model_correct is True][:n_known]
    unknown = [x for x in items if x.model_correct is False][:n_unknown]

    feature_spec = f"L{layer} #{feature_idx}"
    print(f"  intervening on {feature_spec} (scale={scale})  | known={len(known)} unknown={len(unknown)}")

    per_example = []
    baseline_known_correct, ablated_known_correct = 0, 0
    baseline_unknown_correct, ablated_unknown_correct = 0, 0

    def _evaluate(items_subset, group: str):
        nonlocal baseline_known_correct, ablated_known_correct
        nonlocal baseline_unknown_correct, ablated_unknown_correct
        iterator = tqdm(items_subset, desc=f"  {group:7s}", disable=not show_progress)
        for it in iterator:
            # baseline
            ans_base = generate_with_optional_ablation(
                model, tokenizer, it.prompt, sae=None
            )
            correct_base = is_answer_correct(ans_base, it.gold_answers or [])
            # ablation
            ans_abl = generate_with_optional_ablation(
                model, tokenizer, it.prompt,
                sae=sae, layer_idx=layer, feature_idx=feature_idx, scale=scale,
            )
            correct_abl = is_answer_correct(ans_abl, it.gold_answers or [])

            if group == "known":
                baseline_known_correct += int(correct_base)
                ablated_known_correct += int(correct_abl)
            else:
                baseline_unknown_correct += int(correct_base)
                ablated_unknown_correct += int(correct_abl)

            per_example.append({
                "group": group,
                "entity": it.entity,
                "prompt": it.prompt,
                "gold": it.gold_answers,
                "baseline_answer": ans_base,
                "baseline_correct": correct_base,
                "ablated_answer": ans_abl,
                "ablated_correct": correct_abl,
                "flipped": (correct_base != correct_abl),
            })

    _evaluate(known, "known")
    _evaluate(unknown, "unknown")

    base_k = baseline_known_correct / max(len(known), 1)
    abl_k = ablated_known_correct / max(len(known), 1)
    base_u = baseline_unknown_correct / max(len(unknown), 1)
    abl_u = ablated_unknown_correct / max(len(unknown), 1)

    return InterventionResult(
        feature_spec=feature_spec,
        layer=layer,
        feature_idx=feature_idx,
        scale=scale,
        n_known=len(known),
        n_unknown=len(unknown),
        baseline_known_acc=base_k,
        ablated_known_acc=abl_k,
        delta_known=abl_k - base_k,
        baseline_unknown_acc=base_u,
        ablated_unknown_acc=abl_u,
        delta_unknown=abl_u - base_u,
        per_example_changes=per_example,
    )
