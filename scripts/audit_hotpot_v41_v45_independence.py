"""Recompute question- and source-ID overlap between HotpotQA manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def question_ids(rows: list[dict[str, str]]) -> set[str]:
    return {row["pair_id"] for row in rows}


def source_ids(rows: list[dict[str, str]]) -> set[str]:
    values = question_ids(rows)
    for row in rows:
        values.update(value for value in row["distractor_source_ids"].split(";") if value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v41",
        type=Path,
        default=Path("data/manifests/hotpot_v41_source_ids.csv"),
    )
    parser.add_argument(
        "--v45",
        type=Path,
        default=Path("data/manifests/hotpot_v45_source_ids.csv"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    v41 = load_manifest(args.v41)
    v45 = load_manifest(args.v45)
    v41_questions, v45_questions = question_ids(v41), question_ids(v45)
    v41_sources, v45_sources = source_ids(v41), source_ids(v45)
    result = {
        "v41_questions": len(v41_questions),
        "v45_questions": len(v45_questions),
        "question_id_overlap": len(v41_questions & v45_questions),
        "v41_unique_source_ids": len(v41_sources),
        "v45_unique_source_ids": len(v45_sources),
        "source_id_overlap": len(v41_sources & v45_sources),
        "overlapping_question_ids": sorted(v41_questions & v45_questions),
        "overlapping_source_ids": sorted(v41_sources & v45_sources),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
