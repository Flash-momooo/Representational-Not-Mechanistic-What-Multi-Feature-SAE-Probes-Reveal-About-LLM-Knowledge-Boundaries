"""Build an untouched HotpotQA candidate-evidence set for frozen CEVR V42."""

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
from scripts.poc_v41_prepare_hotpot_confirmatory import (
    OPTION_IDS,
    answer_shape,
    candidate_record,
    format_evidence,
)


SOURCE_ID_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])")


def preused_source_ids() -> set[str]:
    """Conservatively exclude every Hotpot-looking source ID in prior data artifacts."""
    used: set[str] = set()
    for path in (ROOT / "data").glob("*.jsonl"):
        used.update(SOURCE_ID_PATTERN.findall(path.read_text(encoding="utf-8")))
    return used


def make_prompt(evidence: str, question: str) -> str:
    return (
        "Use only the evidence below to solve the question. Some sentences "
        "are unrelated distractors. A constrained step will show answer "
        "candidates and request your final choice.\n"
        f"Evidence:\n{evidence}\nQuestion: {question}\nVerification:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", default="data/v42_hotpot_prospective_candidates.jsonl")
    parser.add_argument("--summary-output", default="outputs/poc_v42_hotpot_prepare_results.json")
    parser.add_argument("--protocol", default="paper/CEVR_V42_PROSPECTIVE_PROTOCOL.md")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    dataset = load_dataset(
        "hotpot_qa", "distractor", split="validation",
        cache_dir=str(ROOT / "data" / "cache"),
    )
    excluded = preused_source_ids()
    records = [
        record for row in dataset if (record := candidate_record(row)) is not None
        and record["source_id"] not in excluded
    ]
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        pools[record["answer_shape"]][normalize(record["gold_answer"])].append(record)

    rng = random.Random(args.seed)
    candidates = records.copy()
    rng.shuffle(candidates)
    reserved = set(excluded)
    output = []
    skipped = Counter()
    for record in candidates:
        if record["source_id"] in reserved:
            continue
        answer_norm = normalize(record["gold_answer"])
        groups = [
            key for key, sources in pools[record["answer_shape"]].items()
            if key != answer_norm and any(source["source_id"] not in reserved for source in sources)
        ]
        item_rng = random.Random(f"{args.seed}:{record['source_id']}")
        item_rng.shuffle(groups)
        distractors = []
        for key in groups:
            choices = [source for source in pools[record["answer_shape"]][key]
                       if source["source_id"] not in reserved]
            if choices:
                distractors.append(choices[0])
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            skipped["insufficient_fresh_matched_distractors"] += 1
            continue

        answers = [record["gold_answer"], *[source["gold_answer"] for source in distractors]]
        if len({normalize(answer) for answer in answers}) != 4:
            skipped["duplicate_option"] += 1
            continue
        item_rng.shuffle(answers)
        options = dict(zip(OPTION_IDS, answers))
        gold_option = next(
            option for option, answer in options.items()
            if normalize(answer) == answer_norm
        )
        evidence_pairs = list(record["support"])
        evidence_pairs.extend(source["answer_sentence"] for source in distractors)
        evidence = []
        seen = set()
        for title, sentence in evidence_pairs:
            key = (normalize(title), normalize(sentence))
            if key not in seen:
                evidence.append((title, sentence))
                seen.add(key)
        item_rng.shuffle(evidence)
        evidence_text = format_evidence(evidence)
        evidence_norm = normalize(evidence_text)
        if any(normalize(answer) not in evidence_norm for answer in options.values()):
            skipped["candidate_missing_from_evidence"] += 1
            continue

        output.append({
            "item_id": f"hotpot-v42::{record['source_id']}",
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
            "trajectory_prompt": make_prompt(evidence_text, record["question"]),
            "prompt": make_prompt(evidence_text, record["question"]),
            "distractor_evidence_sources": [source["source_id"] for source in distractors],
            "n_evidence_sentences": len(evidence),
        })
        reserved.add(record["source_id"])
        reserved.update(source["source_id"] for source in distractors)
        if len(output) == args.n_items:
            break

    if len(output) != args.n_items:
        raise ValueError(f"Only built {len(output)} items; requested {args.n_items}")
    source_ids = {row["pair_id"] for row in output}
    distractor_ids = {source for row in output for source in row["distractor_evidence_sources"]}
    if source_ids & excluded or distractor_ids & excluded:
        raise ValueError("V42 source isolation failed")
    if len(source_ids | distractor_ids) != args.n_items * 4:
        raise ValueError("V42 source IDs were reused across items")

    output_path = ROOT / args.output
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    summary = {
        "experiment": "poc_v42_prepare_hotpot_prospective",
        "protocol": args.protocol,
        "n_dataset_rows": len(dataset),
        "n_prior_source_ids_excluded": len(excluded),
        "n_fresh_extractable_records": len(records),
        "n_items": len(output),
        "n_unique_reserved_source_ids": len(source_ids | distractor_ids),
        "answer_shape_counts": dict(Counter(row["answer_shape"] for row in output).most_common()),
        "source_type_counts": dict(Counter(row["source_type"] for row in output).most_common()),
        "gold_option_counts": dict(sorted(Counter(row["gold_option"] for row in output).items())),
        "all_candidate_answers_present_in_evidence": True,
        "no_source_reuse": True,
        "skipped": dict(skipped),
        "output": str(output_path),
    }
    summary_path = ROOT / args.summary_output
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
