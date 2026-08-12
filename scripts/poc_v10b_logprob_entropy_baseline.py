"""PoC V10b: logprob / entropy baselines for SAE-HRM.

This baseline uses the model's own token-level confidence on cached
self-labeled answers:

    prompt + "\nAnswer: " + model_answer

Features:

  - mean answer log-probability
  - minimum answer log-probability
  - final answer token log-probability
  - mean answer entropy
  - maximum answer entropy
  - final answer token entropy
  - answer length

The label is the same as V8-V10:

    hallucination_risk = 1 - model_correct

This is a strong white-box baseline but not an SAE/interpretability method.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from src.data.entity_qa import EntityExample, load_jsonl
from src.load import load_model_and_tokenizer, load_config
from poc_v8_hrm_calibration import expected_calibration_error, fit_platt, triage_curve


DEFAULT_DATA = ROOT / "data" / "popqa_v10_600_self_labeled.jsonl"
DEFAULT_CACHE = ROOT / "outputs" / "cache" / "popqa_v10_600_logprob_entropy_features.npz"
DEFAULT_OUT = ROOT / "outputs" / "poc_v10b_popqa_v10_600_logprob_entropy_results.json"
N_SPLITS = 5
RANDOM_STATE = 42


@dataclass
class BaselineResult:
    method: str
    n: int
    auroc: float
    auprc: float
    f1_at_05: float
    brier: float
    ece_10: float
    mean_predicted_risk: float
    triage: List[dict]
    feature_names: List[str]
    coefficients: List[float]


def build_text(item: EntityExample) -> Tuple[str, int, int]:
    prefix = f"{item.prompt}\nAnswer: "
    answer = item.model_answer or ""
    full = f"{prefix}{answer}".strip()
    return full, len(prefix), len(full)


def answer_token_indices(tokenizer, item: EntityExample) -> Tuple[List[int], str]:
    full, answer_start, answer_end = build_text(item)
    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    token_ids = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if end > answer_start and start < answer_end:
            token_ids.append(idx)
    if not token_ids:
        token_ids = [max(0, len(offsets) - 1)]
    return token_ids, full


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1)


def extract_logprob_entropy_features(model, tokenizer, items: List[EntityExample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_names = [
        "mean_answer_logprob",
        "min_answer_logprob",
        "last_answer_logprob",
        "mean_answer_entropy",
        "max_answer_entropy",
        "last_answer_entropy",
        "answer_token_count",
    ]
    rows = []
    labels_known = []
    device = str(model.device)

    for item in tqdm(items, desc="logprob-entropy"):
        answer_ids, full = answer_token_indices(tokenizer, item)
        inputs = tokenizer(full, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[0]  # (T, vocab)

        logprobs = []
        entropies = []
        for token_idx in answer_ids:
            if token_idx == 0:
                continue
            prev_logits = logits[token_idx - 1]
            target = input_ids[token_idx]
            lp = torch.log_softmax(prev_logits, dim=-1)[target]
            ent = token_entropy(prev_logits)
            logprobs.append(float(lp.detach().cpu()))
            entropies.append(float(ent.detach().cpu()))

        if not logprobs:
            logprobs = [0.0]
            entropies = [0.0]

        rows.append([
            float(np.mean(logprobs)),
            float(np.min(logprobs)),
            float(logprobs[-1]),
            float(np.mean(entropies)),
            float(np.max(entropies)),
            float(entropies[-1]),
            float(len(logprobs)),
        ])
        labels_known.append(1 if item.model_correct else 0)

    return np.array(rows, dtype=np.float64), np.array(labels_known, dtype=np.int64), feature_names


def save_cache(path: Path, X: np.ndarray, labels_known: np.ndarray, feature_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=X,
        labels_known=labels_known,
        feature_names=np.array(feature_names, dtype=object),
    )


def load_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    arr = np.load(path, allow_pickle=True)
    return arr["features"], arr["labels_known"], [str(x) for x in arr["feature_names"]]


def fit_calibrated_baseline(X: np.ndarray, labels_known: np.ndarray, feature_names: List[str]) -> BaselineResult:
    y = 1 - labels_known.astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    coef_sum = np.zeros(X.shape[1], dtype=np.float64)
    coef_count = 0

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        fit_idx, cal_idx = train_test_split(
            train_idx,
            test_size=0.25,
            random_state=RANDOM_STATE + fold,
            stratify=y[train_idx],
        )
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X[fit_idx])
        X_cal = scaler.transform(X[cal_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            clf.fit(X_fit, y[fit_idx])

        cal_scores = clf.predict_proba(X_cal)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        platt = fit_platt(cal_scores, y[cal_idx])
        oof[test_idx] = test_scores if platt is None else platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        coef_sum += clf.coef_.ravel()
        coef_count += 1

    pred = (oof >= 0.5).astype(int)
    return BaselineResult(
        method="logprob_entropy_logreg",
        n=int(len(y)),
        auroc=float(roc_auc_score(y, oof)),
        auprc=float(average_precision_score(y, oof)),
        f1_at_05=float(f1_score(y, pred, zero_division=0)),
        brier=float(brier_score_loss(y, oof)),
        ece_10=float(expected_calibration_error(y, oof, n_bins=10)),
        mean_predicted_risk=float(np.mean(oof)),
        triage=triage_curve(y, oof),
        feature_names=feature_names,
        coefficients=[float(x) for x in (coef_sum / max(coef_count, 1))],
    )


def single_feature_metrics(X: np.ndarray, labels_known: np.ndarray, feature_names: List[str]) -> List[dict]:
    y = 1 - labels_known.astype(int)
    rows = []
    for idx, name in enumerate(feature_names):
        values = X[:, idx]
        auc = roc_auc_score(y, values)
        rows.append({
            "feature": name,
            "signed_auroc_for_risk": float(auc),
            "direction_free_auroc": float(max(auc, 1.0 - auc)),
            "mean_if_hallucinated": float(np.mean(values[y == 1])),
            "mean_if_correct": float(np.mean(values[y == 0])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    cache_path = Path(args.cache)
    out_path = Path(args.out)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}; run V10 first.")

    if cache_path.exists() and not args.force_extract:
        print(f"[v10b] loading cached logprob/entropy features: {cache_path}")
        X, labels_known, feature_names = load_cache(cache_path)
    else:
        print("[v10b] extracting logprob/entropy features")
        cfg = load_config(args.config)
        model, tokenizer = load_model_and_tokenizer(cfg)
        items = load_jsonl(str(data_path))
        X, labels_known, feature_names = extract_logprob_entropy_features(model, tokenizer, items)
        save_cache(cache_path, X, labels_known, feature_names)
        print(f"[v10b] saved cache -> {cache_path}")

    result = fit_calibrated_baseline(X, labels_known, feature_names)
    singles = single_feature_metrics(X, labels_known, feature_names)
    top10 = next(x for x in result.triage if abs(x["route_fraction"] - 0.10) < 1e-9)
    print(
        f"[v10b] AUROC={result.auroc:.4f} AUPRC={result.auprc:.4f} "
        f"Brier={result.brier:.4f} ECE10={result.ece_10:.4f} "
        f"top10_precision={top10['precision_among_routed']:.3f}"
    )
    print("[v10b] single-feature direction-free AUROC:")
    for row in sorted(singles, key=lambda r: -r["direction_free_auroc"]):
        print(f"  {row['feature']}: {row['direction_free_auroc']:.4f}")

    payload = {
        "experiment": "poc_v10b_logprob_entropy_baseline",
        "data": str(data_path),
        "cache": str(cache_path),
        "result": asdict(result),
        "single_feature_metrics": singles,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v10b] saved -> {out_path}")


if __name__ == "__main__":
    main()
