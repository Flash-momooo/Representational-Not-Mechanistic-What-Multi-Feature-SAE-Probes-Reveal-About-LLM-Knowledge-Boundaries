"""V48: 500-question frozen candidate-conditioned dense readout confirmation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.poc_equal_compute_commitment_factorial import (  # noqa: E402
    OPTION_IDS,
    condition_sequences,
    load_npz,
)
from scripts.poc_v40_extract_and_evaluate import padded_batch  # noqa: E402
from src.extract import ResidualHook  # noqa: E402
from src.load import load_config, load_model_and_tokenizer  # noqa: E402

SEED = 20260827
N_BOOTSTRAP = 10_000
LAYER = 18


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def centre(x: np.ndarray, qids: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for qid in np.unique(qids):
        mask = qids == qid
        out[mask] = x[mask].astype(np.float32) - x[mask].astype(np.float32).mean(0, keepdims=True)
    return out


def question_values(y: np.ndarray, score: np.ndarray, qids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aucs, selected = [], []
    for qid in np.unique(qids):
        mask = qids == qid
        aucs.append(float(roc_auc_score(y[mask], score[mask])))
        selected.append(int(y[mask][np.argmin(score[mask])] == 0))
    return np.asarray(aucs), np.asarray(selected)


def bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])], "n_questions": int(len(values))}


def source_model(source: dict[str, np.ndarray]):
    real = source["condition"].astype(str) == "real"
    qids = source["question_ids"].astype(str)[real]
    x = centre(source["raw"][real].astype(np.float32), qids)
    y = source["original_risk"][real].astype(int)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=SEED)
    clf.fit(scaler.transform(x), y)
    return scaler, clf


def extract(items: list[dict], config: str) -> dict[str, np.ndarray]:
    cfg = load_config(config)
    cfg["sae"]["layers"] = [LAYER]
    model, tokenizer = load_model_and_tokenizer(cfg)
    hook = ResidualHook().attach(model.model.layers[LAYER])
    rows = {"raw": [], "label_logprob": [], "risk": [], "question_ids": []}
    try:
        for item in tqdm(items, desc="v48-candidate-readout"):
            sequences, risk, _ = condition_sequences(tokenizer, item, "real")
            ids, mask, lengths = padded_batch(sequences, int(tokenizer.eos_token_id), cfg["model"]["device"])
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
            batch = torch.arange(len(sequences), device=ids.device)
            last = torch.tensor(lengths - 1, device=ids.device)
            prior = torch.tensor(lengths - 2, device=ids.device)
            raw = hook.value[batch, last].float().cpu().numpy().astype(np.float32)
            label_ids = ids[batch, last]
            lp = torch.log_softmax(output.logits[batch, prior].float(), -1)[batch, label_ids]
            rows["raw"].append(raw)
            rows["label_logprob"].append(lp.cpu().numpy().astype(np.float32))
            rows["risk"].append(np.asarray(risk, dtype=np.int8))
            rows["question_ids"].extend([item["item_id"]] * len(OPTION_IDS))
    finally:
        hook.detach()
        del model
        torch.cuda.empty_cache()
    return {"raw": np.concatenate(rows["raw"]), "label_logprob": np.concatenate(rows["label_logprob"]), "risk": np.concatenate(rows["risk"]), "question_ids": np.asarray(rows["question_ids"], dtype=object)}


def evaluate(source: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict:
    qids = target["question_ids"].astype(str)
    y = target["risk"].astype(int)
    scaler, clf = source_model(source)
    dense_risk = clf.predict_proba(scaler.transform(centre(target["raw"], qids)))[:, 1]
    likelihood_risk = -target["label_logprob"].astype(np.float64)
    dense_auc, dense_acc = question_values(y, dense_risk, qids)
    like_auc, like_acc = question_values(y, likelihood_risk, qids)
    return {
        "dense": {"within_question_auroc": bootstrap(dense_auc, SEED + 1), "selection_accuracy": bootstrap(dense_acc, SEED + 2), "population_auroc": float(roc_auc_score(y, dense_risk))},
        "restricted_label_likelihood": {"within_question_auroc": bootstrap(like_auc, SEED + 3), "selection_accuracy": bootstrap(like_acc, SEED + 4), "population_auroc": float(roc_auc_score(y, likelihood_risk))},
        "paired_dense_minus_likelihood": {"within_question_auroc": bootstrap(dense_auc - like_auc, SEED + 5), "selection_accuracy": bootstrap(dense_acc - like_acc, SEED + 6)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fpe5_l18.yaml")
    parser.add_argument("--items", default="data/v47_hotpot_scale_confirmation_candidates.jsonl")
    parser.add_argument("--source-cache", default="outputs/cache/equal_compute_source.npz")
    parser.add_argument("--cache", default="outputs/cache/v48_hotpot_scale_candidate_readout.npz")
    parser.add_argument("--output", default="outputs/poc_v48_hotpot_scale_candidate_readout.json")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    cache = ROOT / args.cache
    if args.force_extract or not cache.exists():
        target = extract(load_items(ROOT / args.items), args.config)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **target)
    else:
        target = load_npz(cache)
    source = load_npz(ROOT / args.source_cache)
    payload = {"experiment": "V48 500-question candidate-conditioned readout", "protocol": "paper/V48_SCALE_CANDIDATE_READOUT_PROTOCOL.md", "n_questions": int(len(np.unique(target["question_ids"].astype(str))),), "n_candidate_states": int(len(target["risk"])), "training": "160-question 2Wiki real-condition source cache only; no target refit", "results": evaluate(source, target), "claim_boundary": "candidate-conditioned readout confirmation, not multi-sample CEVR or pre-generation prediction"}
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
