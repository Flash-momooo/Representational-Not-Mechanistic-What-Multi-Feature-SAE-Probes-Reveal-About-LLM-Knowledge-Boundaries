"""Prepare V40 machine-verifiable factual commitment tasks.

Each item uses a real 2WikiMultiHopQA question and evidence. The gold answer is
mixed with three relation-matched distractors and assigned a randomized option
ID. Later V40 generation will sample a short rationale and then restrict the
final commitment to A/B/C/D, making correctness exactly machine-verifiable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v35_factorial_pilot import build_pairs, normalize


OPTION_IDS = ("A", "B", "C", "D")


def relation_pools(pairs: list[dict]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        answer = str(pair["gold_answer"]).strip()
        if answer and normalize(answer) not in {normalize(x) for x in pools[pair["relation"]]}:
            pools[pair["relation"]].append(answer)
    return pools


def choose_distractors(
    pair: dict,
    pools: dict[str, list[str]],
    global_answers: list[str],
    rng: random.Random,
) -> list[str]:
    gold_norm = normalize(pair["gold_answer"])
    same_relation = [
        answer for answer in pools[pair["relation"]]
        if normalize(answer) != gold_norm
    ]
    rng.shuffle(same_relation)
    selected = same_relation[:3]
    if len(selected) < 3:
        fallback = [
            answer for answer in global_answers
            if normalize(answer) != gold_norm
            and normalize(answer) not in {normalize(x) for x in selected}
        ]
        rng.shuffle(fallback)
        selected.extend(fallback[: 3 - len(selected)])
    if len(selected) != 3:
        raise ValueError(f"Insufficient distractors for {pair['pair_id']}")
    return selected


def format_facts(evidences: list) -> str:
    return "\n".join(
        f"{index}. {str(subject).strip()} | {str(relation).strip()} | {str(obj).strip()}"
        for index, (subject, relation, obj) in enumerate(evidences, 1)
    )


def make_item(
    pair: dict,
    pools: dict[str, list[str]],
    global_answers: list[str],
    seed: int,
) -> dict:
    rng = random.Random(f"{seed}:{pair['pair_id']}")
    options = [pair["gold_answer"], *choose_distractors(pair, pools, global_answers, rng)]
    rng.shuffle(options)
    option_map = dict(zip(OPTION_IDS, options))
    gold_option = next(option for option, answer in option_map.items() if normalize(answer) == normalize(pair["gold_answer"]))
    facts = format_facts(pair["evidences"])
    option_text = "\n".join(f"{option}. {option_map[option]}" for option in OPTION_IDS)
    trajectory_prompt = (
        "Use only the facts below to solve the question. Write one short "
        "verification phrase. Do not use the word FINAL. A separate constrained "
        "step will show answer options and request your final choice.\n"
        f"Facts:\n{facts}\nQuestion: {pair['complex_question']}\nVerification:"
    )
    return {
        "item_id": f"{pair['pair_id']}::complex",
        "pair_id": pair["pair_id"],
        "source_type": pair["source_type"],
        "relation": pair["relation"],
        "question": pair["complex_question"],
        "facts": facts,
        "gold_answer": pair["gold_answer"],
        "options": option_map,
        "options_text": option_text,
        "gold_option": gold_option,
        "trajectory_prompt": trajectory_prompt,
        "prompt": trajectory_prompt,
    }


def validate(items: list[dict]) -> dict:
    for item in items:
        if set(item["options"]) != set(OPTION_IDS):
            raise ValueError(f"Bad option IDs: {item['item_id']}")
        normalized = [normalize(answer) for answer in item["options"].values()]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Duplicate normalized options: {item['item_id']}")
        if normalize(item["options"][item["gold_option"]]) != normalize(item["gold_answer"]):
            raise ValueError(f"Gold option mismatch: {item['item_id']}")
    return {
        "n_items": len(items),
        "gold_option_counts": dict(sorted(Counter(item["gold_option"] for item in items).items())),
        "relation_counts": dict(Counter(item["relation"] for item in items).most_common()),
        "source_type_counts": dict(Counter(item["source_type"] for item in items).most_common()),
        "all_options_unique_after_normalization": True,
        "all_gold_options_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", default="data/v40_machine_verifiable_candidates.jsonl")
    args = parser.parse_args()

    # Request a larger pool so relation-matched distractors can be drawn before
    # truncating to the final confirmatory set.
    pool_size = max(args.n_items * 3, 480)
    pairs = build_pairs(pool_size, args.seed)
    if len(pairs) < args.n_items:
        raise ValueError(f"Only {len(pairs)} eligible pairs, need {args.n_items}")
    pools = relation_pools(pairs)
    global_answers = list(dict.fromkeys(str(pair["gold_answer"]).strip() for pair in pairs))
    items = [make_item(pair, pools, global_answers, args.seed) for pair in pairs[: args.n_items]]
    stats = validate(items)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary_path = ROOT / "outputs" / "poc_v40_machine_verifiable_prepare_results.json"
    summary_path.write_text(
        json.dumps({
            "experiment": "poc_v40_prepare_machine_verifiable",
            "dataset": "2WikiMultiHopQA validation",
            "seed": args.seed,
            "output": str(output_path),
            **stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"saved -> {output_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
