"""Prepare the frozen V41 HotpotQA machine-verifiable confirmation set."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_v35_factorial_pilot import normalize


OPTION_IDS = ("A", "B", "C", "D")
MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def answer_shape(answer: str) -> str:
    text = answer.strip()
    normalized = normalize(text)
    tokens = normalized.split()
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", normalized):
        return "year"
    if any(month in tokens for month in MONTHS):
        return "date"
    if re.search(r"\d", text):
        return "numeric"
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", text)
    capitalized = bool(raw_tokens) and all(token[0].isupper() for token in raw_tokens)
    if len(tokens) == 1 and capitalized:
        return "capitalized_one"
    if 2 <= len(tokens) <= 5 and capitalized:
        return "capitalized_multi"
    if len(tokens) == 1:
        return "lowercase_one"
    if 2 <= len(tokens) <= 4:
        return "lowercase_short"
    return "other_phrase"


def supporting_sentences(row: dict) -> list[tuple[str, str]]:
    context = {
        title: sentences
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
    }
    output = []
    seen = set()
    for title, sentence_id in zip(
        row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
    ):
        sentences = context.get(title, [])
        if sentence_id >= len(sentences):
            continue
        sentence = str(sentences[sentence_id]).strip()
        key = (title, sentence)
        if sentence and key not in seen:
            output.append(key)
            seen.add(key)
    return output


def candidate_record(row: dict) -> dict | None:
    answer = str(row["answer"]).strip()
    if not answer or answer.casefold() in {"yes", "no"}:
        return None
    support = supporting_sentences(row)
    answer_norm = normalize(answer)
    matching = [
        (title, sentence) for title, sentence in support
        if answer_norm and answer_norm in normalize(sentence)
    ]
    if not matching:
        return None
    return {
        "source_id": str(row["id"]),
        "question": str(row["question"]).strip(),
        "gold_answer": answer,
        "answer_shape": answer_shape(answer),
        "hotpot_type": str(row["type"]),
        "level": str(row["level"]),
        "support": support,
        "answer_sentence": matching[0],
    }


def format_evidence(evidence: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{index}. [{title}] {sentence}"
        for index, (title, sentence) in enumerate(evidence, 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", default="data/v41_hotpot_confirmatory_candidates.jsonl")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    dataset = load_dataset(
        "hotpot_qa", "distractor", split="validation",
        cache_dir=str(ROOT / "data" / "cache"),
    )
    records = [record for row in dataset if (record := candidate_record(row)) is not None]
    pools = defaultdict(lambda: defaultdict(list))
    for record in records:
        pools[record["answer_shape"]][normalize(record["gold_answer"])].append(record)

    rng = random.Random(args.seed)
    candidates = records.copy()
    rng.shuffle(candidates)
    output = []
    skipped = Counter()
    for record in candidates:
        answer_norm = normalize(record["gold_answer"])
        answer_groups = [
            key for key in pools[record["answer_shape"]]
            if key != answer_norm
        ]
        item_rng = random.Random(f"{args.seed}:{record['source_id']}")
        item_rng.shuffle(answer_groups)
        if len(answer_groups) < 3:
            skipped[record["answer_shape"]] += 1
            continue
        distractor_sources = [
            pools[record["answer_shape"]][key][0] for key in answer_groups[:3]
        ]
        option_answers = [
            record["gold_answer"],
            *[source["gold_answer"] for source in distractor_sources],
        ]
        if len({normalize(answer) for answer in option_answers}) != 4:
            skipped["duplicate_option"] += 1
            continue
        item_rng.shuffle(option_answers)
        options = dict(zip(OPTION_IDS, option_answers))
        gold_option = next(
            option for option, answer in options.items()
            if normalize(answer) == answer_norm
        )

        evidence = list(record["support"])
        evidence.extend(source["answer_sentence"] for source in distractor_sources)
        deduplicated = []
        seen = set()
        for title, sentence in evidence:
            key = (normalize(title), normalize(sentence))
            if key not in seen:
                deduplicated.append((title, sentence))
                seen.add(key)
        item_rng.shuffle(deduplicated)
        evidence_text = format_evidence(deduplicated)
        evidence_norm = normalize(evidence_text)
        if any(normalize(answer) not in evidence_norm for answer in options.values()):
            skipped["candidate_missing_from_evidence"] += 1
            continue

        prompt = (
            "Use only the evidence below to solve the question. Some sentences "
            "are unrelated distractors. Before seeing answer candidates, write "
            "only a generic two-to-five-word reasoning-plan tag. Describe "
            "operations, not content, such as: find entity then compare attribute. "
            "Do not copy a name, title, date, number, quoted span, or answer from "
            "the evidence, and do not label a choice.\n"
            f"Evidence:\n{evidence_text}\nQuestion: {record['question']}\nReasoning plan:"
        )
        output.append({
            "item_id": f"hotpot::{record['source_id']}",
            "pair_id": record["source_id"],
            "source_type": record["hotpot_type"],
            "relation": record["answer_shape"],
            "answer_shape": record["answer_shape"],
            "level": record["level"],
            "question": record["question"],
            "facts": evidence_text,
            "gold_answer": record["gold_answer"],
            "options": options,
            "options_text": "\n".join(f"{option}. {options[option]}" for option in OPTION_IDS),
            "gold_option": gold_option,
            "trajectory_prompt": prompt,
            "prompt": prompt,
            "distractor_evidence_sources": [
                source["source_id"] for source in distractor_sources
            ],
            "n_evidence_sentences": len(deduplicated),
        })
        if len(output) >= args.n_items:
            break

    if len(output) < args.n_items:
        raise ValueError(f"Only {len(output)} eligible items; requested {args.n_items}")
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    summary = {
        "experiment": "poc_v41_prepare_hotpot_confirmatory",
        "protocol_frozen_before_generation": True,
        "n_dataset_rows": len(dataset),
        "n_extractable_records": len(records),
        "n_items": len(output),
        "answer_shape_counts": dict(Counter(row["answer_shape"] for row in output).most_common()),
        "hotpot_type_counts": dict(Counter(row["source_type"] for row in output).most_common()),
        "gold_option_counts": dict(sorted(Counter(row["gold_option"] for row in output).items())),
        "evidence_sentence_counts": dict(sorted(Counter(row["n_evidence_sentences"] for row in output).items())),
        "all_candidates_present_in_evidence": True,
        "skipped": dict(skipped),
        "output": str(output_path),
    }
    summary_path = ROOT / "outputs" / "poc_v41_hotpot_prepare_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
