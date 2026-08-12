"""PoC V8: SAE-HRM hallucination-risk monitor.

This experiment reuses the V4 self-labeled PopQA cache and reframes the
existing knowledge-boundary probe as a calibrated hallucination-risk estimator:

    y_risk = 1 - model_correct

For each layer, we train an L1 logistic probe on SAE features with an outer
5-fold split. Inside each training fold, we reserve a calibration split and fit
a one-dimensional Platt calibrator on the probe's validation probabilities.

Outputs:
  - discrimination: AUROC / AUPRC / F1
  - calibration: Brier score / ECE
  - triage utility: how many errors are captured when the highest-risk items
    are routed to retrieval, abstention, or human review
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.entity_qa import load_jsonl
from src.extract import extract_features
from src.load import load_all


LABELED_CACHE = "data/popqa_self_labeled.jsonl"
N_SPLITS = 5
RANDOM_STATE = 42


@dataclass
class RiskProbeResult:
    layer: int
    n: int
    hallucination_rate: float
    auroc: float
    auprc: float
    f1_at_05: float
    brier: float
    ece_10: float
    mean_predicted_risk: float
    fold_aurocs: List[float]
    n_nonzero_per_fold: List[int]
    triage: List[dict]
    threshold_table: List[dict]
    top_features: List[dict]


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(conf - acc)
    return ece


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    """Fit a scalar Platt calibrator.

    Returns None if the calibration split has only one class; the caller should
    then fall back to uncalibrated probabilities.
    """
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        clf.fit(scores.reshape(-1, 1), y)
    return clf


def triage_curve(y_true: np.ndarray, p: np.ndarray, budgets=(0.05, 0.10, 0.20, 0.30, 0.50)) -> List[dict]:
    """Route top-risk items and report captured hallucinations.

    Think of routed examples as sent to RAG, abstention, or human review. Recall
    is the fraction of all hallucinations captured by the route.
    """
    n = len(y_true)
    total_errors = int(np.sum(y_true))
    order = np.argsort(-p)
    rows = []
    for b in budgets:
        k = max(1, int(round(n * b)))
        flagged = order[:k]
        captured = int(np.sum(y_true[flagged]))
        precision = captured / k
        recall = captured / total_errors if total_errors else 0.0
        residual_rate = (total_errors - captured) / n
        rows.append({
            "route_fraction": float(b),
            "n_routed": int(k),
            "captured_hallucinations": captured,
            "precision_among_routed": float(precision),
            "hallucination_recall": float(recall),
            "residual_hallucination_rate_if_fixed": float(residual_rate),
        })
    return rows


def threshold_table(y_true: np.ndarray, p: np.ndarray, thresholds=(0.30, 0.50, 0.70, 0.80, 0.90)) -> List[dict]:
    rows = []
    total_errors = int(np.sum(y_true))
    for t in thresholds:
        pred = p >= t
        n_flagged = int(np.sum(pred))
        if n_flagged == 0:
            precision = recall = f1 = 0.0
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, pred.astype(int), average="binary", zero_division=0
            )
        rows.append({
            "threshold": float(t),
            "n_flagged": n_flagged,
            "flagged_fraction": float(n_flagged / len(y_true)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "captured_hallucinations": int(round(float(recall) * total_errors)),
        })
    return rows


def top_feature_summary(coef_sum: np.ndarray, coef_count: np.ndarray, layer: int, width: str, top_k: int = 20) -> List[dict]:
    avg = np.divide(coef_sum, np.maximum(coef_count, 1), out=np.zeros_like(coef_sum), where=coef_count > 0)
    stable = np.flatnonzero(coef_count >= 3)
    if len(stable) == 0:
        stable = np.flatnonzero(coef_count > 0)
    order = np.argsort(-np.abs(avg[stable]))[:top_k]
    url = "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-{width}/{idx}"
    return [
        {
            "feature_idx": int(j),
            "weight": float(avg[j]),
            "folds_selected": int(coef_count[j]),
            "neuronpedia": url.format(layer=layer, width=width, idx=int(j)),
        }
        for j in stable[order]
    ]


def fit_calibrated_risk_probe(
    features: np.ndarray,
    labels_known: np.ndarray,
    layer: int,
    width: str,
    C: float = 0.1,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> RiskProbeResult:
    y = 1 - labels_known.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_p = np.zeros(len(y), dtype=np.float64)
    fold_aurocs: List[float] = []
    n_nonzero: List[int] = []
    coef_sum = np.zeros(features.shape[1], dtype=np.float64)
    coef_count = np.zeros(features.shape[1], dtype=np.int32)

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, y), start=1):
        X_train_all, X_test = features[train_idx], features[test_idx]
        y_train_all, y_test = y[train_idx], y[test_idx]

        train_fit_idx, train_cal_idx = train_test_split(
            np.arange(len(train_idx)),
            test_size=0.25,
            random_state=random_state + fold,
            stratify=y_train_all,
        )
        X_fit, y_fit = X_train_all[train_fit_idx], y_train_all[train_fit_idx]
        X_cal, y_cal = X_train_all[train_cal_idx], y_train_all[train_cal_idx]

        scaler = StandardScaler(with_mean=False)
        X_fit_s = scaler.fit_transform(X_fit)
        X_cal_s = scaler.transform(X_cal)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            clf.fit(X_fit_s, y_fit)

        cal_scores = clf.predict_proba(X_cal_s)[:, 1]
        test_scores = clf.predict_proba(X_test_s)[:, 1]
        platt = fit_platt(cal_scores, y_cal)
        if platt is None:
            test_p = test_scores
        else:
            test_p = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        oof_p[test_idx] = test_p
        fold_aurocs.append(float(roc_auc_score(y_test, test_p)))

        coef = clf.coef_.ravel()
        nz = np.flatnonzero(coef)
        n_nonzero.append(int(len(nz)))
        coef_sum[nz] += coef[nz]
        coef_count[nz] += 1

    pred_05 = (oof_p >= 0.5).astype(int)
    return RiskProbeResult(
        layer=layer,
        n=int(len(y)),
        hallucination_rate=float(np.mean(y)),
        auroc=float(roc_auc_score(y, oof_p)),
        auprc=float(average_precision_score(y, oof_p)),
        f1_at_05=float(f1_score(y, pred_05, zero_division=0)),
        brier=float(brier_score_loss(y, oof_p)),
        ece_10=float(expected_calibration_error(y, oof_p, n_bins=10)),
        mean_predicted_risk=float(np.mean(oof_p)),
        fold_aurocs=fold_aurocs,
        n_nonzero_per_fold=n_nonzero,
        triage=triage_curve(y, oof_p),
        threshold_table=threshold_table(y, oof_p),
        top_features=top_feature_summary(coef_sum, coef_count, layer, width),
    )


def load_cached_features(path: Path) -> Dict[int, dict]:
    arr = np.load(path, allow_pickle=True)
    layers = [int(x) for x in arr["layers"]]
    out = {}
    for layer in layers:
        out[layer] = {
            "features": arr[f"features_L{layer}"],
            "labels": arr["labels"],
        }
    return out


def save_cached_features(path: Path, feats: Dict[int, object]) -> None:
    payload = {"layers": np.array(sorted(feats.keys()), dtype=np.int64)}
    first = next(iter(feats.values()))
    payload["labels"] = first.labels
    for layer, fb in feats.items():
        payload[f"features_L{layer}"] = fb.features
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=LABELED_CACHE)
    parser.add_argument("--cache", default="outputs/cache/poc_v8_self_labeled_sae_features.npz")
    parser.add_argument("--out", default="outputs/poc_v8_hrm_calibration_results.json")
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}; run scripts/poc_v4_self_knowledge.py first.")

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.force_extract:
        print(f"[v8] loading cached SAE features: {cache_path}")
        cached = load_cached_features(cache_path)
        layers = sorted(cached.keys())
        width = "16k"
    else:
        print("[v8] extracting SAE features from self-labeled PopQA")
        assets = load_all(args.config)
        items = load_jsonl(str(data_path))
        feats = extract_features(assets, items)
        save_cached_features(cache_path, feats)
        print(f"[v8] saved feature cache -> {cache_path}")
        cached = {L: {"features": fb.features, "labels": fb.labels} for L, fb in feats.items()}
        layers = sorted(cached.keys())
        width = assets.cfg["sae"]["width"]

    results = []
    for layer in layers:
        print(f"\n[v8] fitting calibrated hallucination-risk probe: L{layer}")
        res = fit_calibrated_risk_probe(
            cached[layer]["features"],
            cached[layer]["labels"],
            layer=layer,
            width=width,
            C=args.C,
        )
        results.append(asdict(res))
        print(
            f"  AUROC={res.auroc:.4f}  AUPRC={res.auprc:.4f}  "
            f"Brier={res.brier:.4f}  ECE10={res.ece_10:.4f}"
        )
        top10 = next(x for x in res.triage if abs(x["route_fraction"] - 0.10) < 1e-9)
        print(
            "  route top 10%: "
            f"precision={top10['precision_among_routed']:.3f}, "
            f"hallucination recall={top10['hallucination_recall']:.3f}"
        )

    best = max(results, key=lambda x: (x["auroc"], -x["ece_10"]))
    payload = {
        "experiment": "poc_v8_hrm_calibration",
        "label_definition": "hallucination_risk = 1 - model_correct from V4 self-evaluation labels",
        "notes": [
            "Outer folds are held out from both probe fitting and Platt calibration.",
            "Triage assumes routed items would be handled by RAG, abstention, or human review.",
            "This validates monitoring utility, not a causal hallucination mechanism.",
        ],
        "best_layer_by_auroc": best["layer"],
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[v8] saved -> {out_path}")


if __name__ == "__main__":
    main()
