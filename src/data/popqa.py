"""PopQA 真实实体数据集加载器（Mallen et al. 2023 EMNLP）。

每条 PopQA 包含一个实体 (subj) 与该实体某属性的问题，附带 Wikipedia 月浏览量
作为流行度（popularity）。我们按浏览量将实体分为：
  - popular（top quantile）→ 候选 known
  - obscure（bottom quantile）→ 候选 unknown

相比 Ferrando 论文用的小数据，PopQA 的优势：
  1. 实体都是**真实存在**的（避免我们 toy 数据 AUROC=1.0 的 trivial 问题）
  2. 浏览量提供**连续 popularity 信号**，可调难度
  3. 完全开源在 HF: `akariasai/PopQA`

API 与 src.data.entity_qa 对齐（返回 EntityExample 列表），可无缝替换。
"""
from __future__ import annotations

import os
from typing import List, Optional

from .entity_qa import EntityExample


PROPERTY_TO_TYPE = {
    # 把 PopQA 的 22 种 property 大致映射到 entity_type，方便后续按类型分析
    "occupation": "person",
    "place_of_birth": "person",
    "genre": "creative_work",
    "father": "person",
    "country": "place",
    "producer": "creative_work",
    "director": "creative_work",
    "capital_of": "place",
    "screenwriter": "creative_work",
    "composer": "creative_work",
    "color": "object",
    "religion": "person",
    "sport": "person",
    "author": "creative_work",
    "mother": "person",
    "capital": "place",
}


def load_popqa(
    split: str = "test",
    n_popular: int = 200,
    n_obscure: int = 200,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[EntityExample]:
    """从 HF 下载 PopQA，按 s_pop 分层取 popular/obscure。

    Args:
        split: PopQA 没有 train/test 划分，传 "test" 即取全部
        n_popular: 取 top 流行度的 N 条作为 known 候选
        n_obscure: 取 bottom 流行度的 N 条作为 unknown 候选
        cache_dir: HF datasets 缓存目录
        seed: 随机种子（用于打乱）

    Returns:
        EntityExample 列表，label=1 表示 popular（候选 known），label=0 表示 obscure
    """
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split=split, cache_dir=cache_dir)
    # 按 s_pop 排序（s_pop 越大越流行）
    rows = list(ds)
    # 兼容 PopQA 字段：HF 上是 's_pop'，已是数值
    rows.sort(key=lambda r: r.get("s_pop") or 0, reverse=True)

    popular = rows[:n_popular]
    obscure = rows[-n_obscure:] if n_obscure > 0 else []

    items: List[EntityExample] = []

    def _to_example(r: dict, label: int) -> EntityExample:
        subj = r["subj"]
        prop = r.get("prop", "")
        # 直接用 PopQA 给的 question 字段
        question = r["question"]
        etype = PROPERTY_TO_TYPE.get(prop, "entity")
        # PopQA 的 possible_answers 是 JSON 字符串如 '["computer programmer", "writer"]'
        import ast
        raw = r.get("possible_answers")
        if isinstance(raw, str):
            try:
                gold = ast.literal_eval(raw)
                if not isinstance(gold, list):
                    gold = [str(gold)]
            except (ValueError, SyntaxError):
                gold = [raw]
        elif isinstance(raw, list):
            gold = raw
        else:
            gold = None
        return EntityExample(entity=subj, entity_type=etype,
                             prompt=question, label=label,
                             gold_answers=gold)

    for r in popular:
        items.append(_to_example(r, label=1))
    for r in obscure:
        items.append(_to_example(r, label=0))

    # 打乱顺序，避免训练时偏序
    import random
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def get_or_build_popqa(
    cache_path: str = "data/popqa_subset.jsonl",
    n_popular: int = 200,
    n_obscure: int = 200,
    force_rebuild: bool = False,
) -> List[EntityExample]:
    """带本地缓存版本：第一次跑会从 HF 下载并截断保存到 jsonl。"""
    from .entity_qa import save_jsonl, load_jsonl
    if (not force_rebuild) and os.path.exists(cache_path):
        return load_jsonl(cache_path)
    items = load_popqa(n_popular=n_popular, n_obscure=n_obscure)
    save_jsonl(items, cache_path)
    print(f"[popqa] saved {len(items)} examples -> {cache_path}")
    return items


if __name__ == "__main__":
    items = get_or_build_popqa(n_popular=50, n_obscure=50)
    n_pop = sum(1 for x in items if x.label == 1)
    print(f"total={len(items)}  popular={n_pop}  obscure={len(items)-n_pop}")
    print("\n[popular sample]:", next(x for x in items if x.label == 1))
    print("[obscure sample]:", next(x for x in items if x.label == 0))
