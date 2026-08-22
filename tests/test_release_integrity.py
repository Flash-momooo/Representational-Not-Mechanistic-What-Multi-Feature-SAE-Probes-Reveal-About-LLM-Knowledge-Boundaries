"""CPU-only integrity checks for the publication release."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    forbidden_suffixes = {".docx", ".tex", ".aux", ".bbl", ".blg", ".png", ".jpg", ".svg"}
    forbidden_names = {"paper", "figures", "figure_versions", "models", "checkpoints", "tools"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        assert not forbidden_names.intersection(relative.parts), relative
        if path.is_file():
            assert path.suffix.lower() not in forbidden_suffixes, relative

    result = json.loads((ROOT / "results" / "frozen_cevr_confirmation.json").read_text(encoding="utf-8"))
    assert result["target"]["source_isolation_audit"]["passes_zero_overlap"] is True
    assert result["t0_collision_control"]["within_question_auroc"] == 0.5
    assert result["results"]["first_sample"]["accuracy"] == 0.3375
    assert result["results"]["frozen_rank32_listwise_adapter"]["accuracy"] == 0.69375

    text_result = json.loads(
        (ROOT / "results" / "finetuned_text_cross_encoder.json").read_text(encoding="utf-8")
    )
    assert text_result["target"]["n_questions"] == 160
    assert text_result["result"]["accuracy"] == 0.40625

    stage_result = json.loads(
        (ROOT / "results" / "stage_budget_intervention.json").read_text(encoding="utf-8")
    )
    assert stage_result["experiment"] == "NN36 stage-wise budget-constrained intervention"

    with (ROOT / "data" / "ARTIFACT_MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["sha256"] and len(row["sha256"]) == 64 for row in rows)
    print(f"release integrity OK: {len(rows)} external artifacts indexed")


if __name__ == "__main__":
    main()
