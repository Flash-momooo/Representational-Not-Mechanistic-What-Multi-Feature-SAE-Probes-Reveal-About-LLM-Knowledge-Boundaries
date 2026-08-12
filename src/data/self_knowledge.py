"""V4：用模型 self-evaluation 给 PopQA 重新打标签。

核心思想：用 Gemma 实际回答 PopQA 问题，对照 gold answers 判定 known/unknown：
  - label = 1 (known)  : 模型回答包含任一 gold answer
  - label = 0 (unknown): 模型回答与所有 gold answers 都不匹配（错误/refuse/幻觉）

相比 V3 用 popularity 启发式打标，这种"模型实际行为"标签：
  1. 直接回答 reviewer 的质疑（"是 popularity 还是 knowledge"）
  2. 剔除流行但模型不会的样本（污染 known 类）
  3. 剔除冷门但模型刚好会的样本（污染 unknown 类）
  4. 与 Ferrando §3 的"模型自评估"思路一致
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from .entity_qa import EntityExample


# Few-shot prompt：用 2 个简单示例引导模型给短答案
FEW_SHOT_TEMPLATE = """Q: What is the capital of France?
A: Paris

Q: Who wrote "Romeo and Juliet"?
A: William Shakespeare

Q: {question}
A:"""


def _normalize(s: str) -> str:
    """规范化：小写 + 去标点 + 折叠空白。"""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_answer_correct(response: str, gold_answers: List[str]) -> bool:
    """response 包含任一 gold answer 即判正确（子串匹配，规范化后）。"""
    if not gold_answers:
        return False
    r = _normalize(response)
    for g in gold_answers:
        g_norm = _normalize(g)
        if g_norm and g_norm in r:
            return True
    return False


def _generate_answer(model, tokenizer, question: str,
                     max_new_tokens: int = 20, device: str = "cuda") -> str:
    """单条生成。返回 A: 后面到下一个 Q: 或换行为止的字符串。"""
    prompt = FEW_SHOT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # greedy，可复现
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # 截断到第一个换行或 "Q:"（避免模型继续生成下一个问题）
    for stop in ["\nQ:", "\n\n", "\n"]:
        if stop in text:
            text = text.split(stop)[0]
            break
    return text.strip()


def label_with_self_knowledge(
    model,
    tokenizer,
    items: List[EntityExample],
    max_new_tokens: int = 20,
    device: str = "cuda",
    show_progress: bool = True,
) -> List[EntityExample]:
    """对每个 item 调用模型回答、判定对错、写入 model_answer / model_correct / label。

    注：返回新的列表（不修改输入对象的内存），label 被改为模型实际表现。
    """
    out: List[EntityExample] = []
    iterator = tqdm(items, desc="self-eval", disable=not show_progress)
    for it in iterator:
        if it.gold_answers is None:
            # 没有 gold 答案的样本，原样保留并标记 None
            out.append(EntityExample(
                entity=it.entity, entity_type=it.entity_type,
                prompt=it.prompt, label=it.label,
                gold_answers=it.gold_answers,
                model_answer=None, model_correct=None,
            ))
            continue
        ans = _generate_answer(model, tokenizer, it.prompt,
                               max_new_tokens=max_new_tokens, device=device)
        correct = is_answer_correct(ans, it.gold_answers)
        out.append(EntityExample(
            entity=it.entity, entity_type=it.entity_type,
            prompt=it.prompt,
            label=1 if correct else 0,        # 新标签 = 模型实际表现
            gold_answers=it.gold_answers,
            model_answer=ans,
            model_correct=correct,
        ))
    return out


def summarize(items: List[EntityExample]) -> dict:
    """统计标注结果。"""
    total = len(items)
    labeled = [x for x in items if x.model_correct is not None]
    n_correct = sum(1 for x in labeled if x.model_correct)
    # 与原 popularity 标签的关系（如果有的话）
    return {
        "total": total,
        "labeled": len(labeled),
        "model_correct": n_correct,
        "model_wrong": len(labeled) - n_correct,
        "accuracy": n_correct / max(len(labeled), 1),
    }
