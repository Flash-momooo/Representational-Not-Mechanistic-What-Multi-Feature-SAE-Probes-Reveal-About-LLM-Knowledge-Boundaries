"""Ferrando 风格的 entity 知识数据集（PoC 用小规模版本）。

每条样本：
    {
        "entity": "Inception",
        "entity_type": "movie",
        "prompt": "Tell me a fact about the movie Inception.",
        "label": 1   # 1 = known (popular)，0 = unknown (fictional / obscure)
    }

PoC 阶段：手工构造 ~200 条混合 known/unknown 样本，
后续可扩展到完整 Ferrando 数据 / TriviaQA。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EntityExample:
    entity: str
    entity_type: str
    prompt: str
    label: int          # 1 = known, 0 = unknown
    # 可选：gold 答案列表（PopQA 等真实数据集才有；用于 V4 self-knowledge labeling）
    gold_answers: Optional[List[str]] = None
    # 可选：模型生成的回答 + 是否答对（V4 标注后填入）
    model_answer: Optional[str] = None
    model_correct: Optional[bool] = None


# ---- 一组 demo 种子（足够跑通 PoC；后续替换为更大数据集）----
KNOWN_ENTITIES = {
    "movie": ["Inception", "The Godfather", "Titanic", "Avatar", "Pulp Fiction",
              "The Dark Knight", "Forrest Gump", "Interstellar", "Fight Club", "Gladiator"],
    "city": ["Paris", "Tokyo", "New York", "London", "Beijing",
             "Sydney", "Berlin", "Rome", "Cairo", "Moscow"],
    "player": ["Lionel Messi", "Cristiano Ronaldo", "LeBron James", "Michael Jordan",
               "Roger Federer", "Serena Williams", "Tom Brady", "Tiger Woods"],
    "song": ["Bohemian Rhapsody", "Imagine", "Hey Jude", "Yesterday",
             "Hotel California", "Stairway to Heaven", "Smells Like Teen Spirit"],
}

# Unknown：刻意构造的虚构实体（plausible 但不存在）
UNKNOWN_ENTITIES = {
    "movie": ["The Crimson Veil of Zantar", "Echoes of the Forgotten Atlas",
              "Whispering Halls of Brindlemoor", "Glass Children of Verothil",
              "The Last Cartographer of Yssel", "Moonlit Fugue in B-flat",
              "Saltgrave Recursion", "The Eleventh Doorway", "Pale Equinox",
              "Hollow Tides of Marivell"],
    "city": ["Vrothendale", "Quenmoria", "Sablecreek-on-Hest", "Briartown Hollow",
             "Pellumbra City", "North Mardrigal", "Ostvellin", "Krynne Cove",
             "Tarsivet", "Eldenmark Bay"],
    "player": ["Marcus Velthrane", "Yiren Kasparov", "Solomon Drestige",
               "Paolo Vincentini Junior", "Akemi Hoshino-Reis", "Felix Brombaugh",
               "Niamh O'Cleirigh", "Damir Vakhtangov"],
    "song": ["Glasshouse Reverie", "October's Long Spine", "Vermillion Static",
             "Winter Oar in Minor", "Hush of the Twelve Stations",
             "The Cartographer's Lament", "Brittle Sun"],
}

PROMPT_TEMPLATES = {
    "movie": "Tell me a fact about the movie {e}.",
    "city":  "Tell me a fact about the city of {e}.",
    "player":"Tell me a fact about the athlete {e}.",
    "song":  "Tell me a fact about the song {e}.",
}


def build_dataset(seed: int = 42, max_per_class: int | None = None) -> List[EntityExample]:
    rng = random.Random(seed)
    items: List[EntityExample] = []
    for et in KNOWN_ENTITIES.keys():
        kn = KNOWN_ENTITIES[et]
        un = UNKNOWN_ENTITIES[et]
        if max_per_class is not None:
            kn = rng.sample(kn, k=min(max_per_class, len(kn)))
            un = rng.sample(un, k=min(max_per_class, len(un)))
        for e in kn:
            items.append(EntityExample(e, et, PROMPT_TEMPLATES[et].format(e=e), label=1))
        for e in un:
            items.append(EntityExample(e, et, PROMPT_TEMPLATES[et].format(e=e), label=0))
    rng.shuffle(items)
    return items


def save_jsonl(items: List[EntityExample], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[EntityExample]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(EntityExample(**d))
    return items


def get_or_build(path: str = "data/entity_qa.jsonl") -> List[EntityExample]:
    if os.path.exists(path):
        return load_jsonl(path)
    items = build_dataset()
    save_jsonl(items, path)
    print(f"[data] built {len(items)} entity examples -> {path}")
    return items


if __name__ == "__main__":
    items = get_or_build()
    n_pos = sum(1 for x in items if x.label == 1)
    print(f"total={len(items)}  known={n_pos}  unknown={len(items)-n_pos}")
    print("sample:", items[0])
