"""Aggregate V48 and V49 candidate verifiers on one fixed target table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import load_npz  # noqa: E402
from scripts.poc_v48_scale_candidate_readout import centre, source_model  # noqa: E402


SEED = 20260827
N_BOOTSTRAP = 10_000


def bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])], "n_questions": int(len(values))}


def candidate_metrics(correct: np.ndarray, qids: np.ndarray, utility: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    question_aucs, selected = [], []
    for qid in np.unique(qids):
        indices = np.flatnonzero(qids == qid)
        question_aucs.append(float(roc_auc_score(correct[indices], utility[indices])))
        selected.append(int(indices[np.argmax(utility[indices])]))
    selected = np.asarray(selected, dtype=int)
    return {
        "within_question_auroc": bootstrap(np.asarray(question_aucs), seed),
        "selection_accuracy": bootstrap(correct[selected], seed + 1),
        "population_auroc": float(roc_auc_score(correct, utility)),
        "population_auprc": float(average_precision_score(correct, utility)),
        "selected_indices": [int(i) for i in selected],
    }, selected


def from_v49(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name, record = next(iter(payload["methods"].items()))
    return name, record["target"]


def main() -> None:
    target = load_npz(ROOT / "outputs/cache/v48_hotpot_scale_candidate_readout.npz")
    source = load_npz(ROOT / "outputs/cache/equal_compute_source.npz")
    qids = target["question_ids"].astype(str)
    correct = 1 - target["risk"].astype(int)
    scaler, clf = source_model(source)
    dense_risk = clf.predict_proba(scaler.transform(centre(target["raw"], qids)))[:, 1]
    methods: dict[str, dict] = {}
    _, dense_choice = candidate_metrics(correct, qids, -dense_risk, SEED + 1)
    methods["frozen_dense_state_readout"] = candidate_metrics(correct, qids, -dense_risk, SEED + 1)[0]
    _, likelihood_choice = candidate_metrics(correct, qids, target["label_logprob"].astype(float), SEED + 2)
    methods["restricted_label_likelihood"] = candidate_metrics(correct, qids, target["label_logprob"].astype(float), SEED + 2)[0]
    for filename in (
        "outputs/poc_v49_frozen_text_candidate_verifier.json",
        "outputs/poc_v49_finetuned_text_candidate_verifier.json",
        "outputs/poc_v49_zero_shot_qwen7b_candidate_verifier.json",
    ):
        name, record = from_v49(ROOT / filename)
        methods[name] = record
    dense_success = correct[dense_choice]
    comparisons = {}
    for name, record in methods.items():
        if name == "frozen_dense_state_readout":
            continue
        challenger_success = correct[np.asarray(record["selected_indices"], dtype=int)]
        comparisons["dense_minus_" + name] = bootstrap(dense_success - challenger_success, SEED + len(comparisons) + 20)
    output = {
        "experiment": "V49 aggregate strong candidate-verifier audit",
        "protocol": "paper/V49_STRONG_CANDIDATE_VERIFIER_AUDIT_PROTOCOL.md",
        "status": "post-hoc comparison on the already examined V48 target; all V48 state scores remain frozen",
        "target": {"n_questions": int(len(np.unique(qids))), "n_candidate_states": int(len(qids)), "candidate_pool": "four supplied options per question"},
        "methods": methods,
        "paired_selection_accuracy_dense_minus": comparisons,
        "interpretation": "A zero-shot 7B external evidence verifier is stronger than the frozen 2B internal state readout under this fully evidence-grounded finite-option task. This rules out a universal internal-readout superiority claim; it does not change the exact-prefix observability bound.",
    }
    path = ROOT / "outputs/poc_v49_strong_candidate_verifier_aggregate.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
