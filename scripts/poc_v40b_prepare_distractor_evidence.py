"""V40b: add real relation-matched evidence for every distractor candidate."""

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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def triple_key(triple) -> tuple[str, str, str]:
    return tuple(normalize(str(value)) for value in triple[:3])


def format_facts(triples: list) -> str:
    return "\n".join(
        f"{index}. {str(subject).strip()} | {str(relation).strip()} | {str(obj).strip()}"
        for index, (subject, relation, obj) in enumerate(triples, 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/v40_machine_verifiable_candidates.jsonl")
    parser.add_argument("--output", default="data/v40b_distractor_evidence_candidates.jsonl")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    items = load_jsonl(ROOT / args.input)
    pairs = build_pairs(5000, args.seed)
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    relation_sources: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        relation_sources[pair["relation"]][normalize(pair["gold_answer"])].append(pair)

    output_rows = []
    skipped = []
    for item in items:
        current = pair_by_id[item["pair_id"]]
        triples = [list(triple[:3]) for triple in current["evidences"]]
        gold_norm = normalize(current["gold_answer"])
        source_groups = relation_sources[current["relation"]]
        available_answers = [answer for answer in source_groups if answer != gold_norm]
        rng = random.Random(f"{args.seed}:{item['item_id']}:distractor-evidence")
        rng.shuffle(available_answers)
        if len(available_answers) < 3:
            skipped.append({
                "item_id": item["item_id"],
                "relation": current["relation"],
                "available_distractors": len(available_answers),
            })
            continue

        selected_sources = [source_groups[answer][0] for answer in available_answers[:3]]
        option_answers = [current["gold_answer"], *[source["gold_answer"] for source in selected_sources]]
        rng.shuffle(option_answers)
        option_map = dict(zip(OPTION_IDS, option_answers))
        gold_option = next(
            option for option, answer in option_map.items()
            if normalize(answer) == gold_norm
        )
        options_text = "\n".join(f"{option}. {option_map[option]}" for option in OPTION_IDS)

        distractor_sources = {}
        source_by_answer = {normalize(source["gold_answer"]): source for source in selected_sources}
        for option, answer in option_map.items():
            if option == gold_option:
                continue
            source = source_by_answer[normalize(answer)]
            triple = list(source["evidences"][-1][:3])
            if normalize(str(triple[2])) != normalize(answer):
                raise ValueError(f"Distractor object mismatch: {item['item_id']} {option}")
            triples.append(triple)
            distractor_sources[option] = source["pair_id"]

        deduplicated = []
        seen = set()
        for triple in triples:
            key = triple_key(triple)
            if key not in seen:
                deduplicated.append(triple)
                seen.add(key)
        rng.shuffle(deduplicated)
        facts = format_facts(deduplicated)
        trajectory_prompt = (
            "Use only the facts below to solve the question. Some facts are "
            "unrelated distractors. Write one short verification phrase. Do not "
            "use the word FINAL. A separate constrained step will show answer "
            "candidates and request your final choice.\n"
            f"Facts:\n{facts}\nQuestion: {item['question']}\nVerification:"
        )
        row = {
            **item,
            "options": option_map,
            "options_text": options_text,
            "gold_option": gold_option,
            "facts": facts,
            "trajectory_prompt": trajectory_prompt,
            "prompt": trajectory_prompt,
            "distractor_evidence_sources": distractor_sources,
            "n_facts": len(deduplicated),
        }
        object_values = {normalize(str(triple[2])) for triple in deduplicated}
        for option, answer in option_map.items():
            if normalize(answer) not in object_values:
                raise ValueError(f"Option missing from evidence objects: {item['item_id']} {option}")
        output_rows.append(row)

    output_path = ROOT / args.output
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    summary = {
        "experiment": "poc_v40b_prepare_distractor_evidence",
        "n_input_items": len(items),
        "n_items": len(output_rows),
        "n_skipped": len(skipped),
        "skipped_relation_counts": dict(Counter(row["relation"] for row in skipped).most_common()),
        "gold_option_counts": dict(sorted(Counter(row["gold_option"] for row in output_rows).items())),
        "relation_counts": dict(Counter(row["relation"] for row in output_rows).most_common()),
        "mean_facts": sum(row["n_facts"] for row in output_rows) / len(output_rows),
        "fact_count_distribution": dict(sorted(Counter(row["n_facts"] for row in output_rows).items())),
        "all_candidates_present_as_evidence_objects": True,
        "facts_randomized_by_item": True,
        "output": str(output_path),
    }
    summary_path = ROOT / "outputs" / "poc_v40b_prepare_distractor_evidence_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
