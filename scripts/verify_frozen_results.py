"""Verify headline claims directly from released machine-readable records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, observed {actual}")


def main() -> None:
    payload = json.loads((ROOT / "results" / "frozen_cevr_confirmation.json").read_text(encoding="utf-8"))
    first = payload["results"]["first_sample"]
    adapter = payload["results"]["frozen_rank32_listwise_adapter"]
    close(first["accuracy"], 0.3375)
    close(adapter["accuracy"], 0.69375)
    close(adapter["accuracy"] - first["accuracy"], 0.35625)
    interval = adapter["paired_vs_first"]["ci95"]
    assert interval[0] > 0.0 and interval[0] <= interval[1]
    assert payload["target"]["source_isolation_audit"]["n_overlapping_prior_source_ids"] == 0
    close(payload["t0_collision_control"]["within_question_auroc"], 0.5)
    print("frozen headline results verified")


if __name__ == "__main__":
    main()

