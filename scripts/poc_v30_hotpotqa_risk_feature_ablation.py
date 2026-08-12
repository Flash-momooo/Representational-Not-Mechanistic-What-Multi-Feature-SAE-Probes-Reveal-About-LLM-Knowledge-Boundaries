"""PoC V30: hallucination-risk SAE feature ablation on HotpotQA.

This experiment extends the paper's "representational, not mechanistic" test
from entity knowledge boundaries to answer-time hallucination-risk monitoring.

Question:
  Do SAE features selected by a HotpotQA answer-time risk probe causally control
  factual retrieval, or are they mainly representational risk markers?

Protocol:
  1. Load cached HotpotQA answer-position SAE features.
  2. Train sparse risk probes on answer_first L20 features.
  3. Select top positive SAE features for hallucination risk.
  4. Jointly ablate the selected features and matched random active features.
  5. Measure delta log P(gold answer | prompt) on correct and wrong examples.

The desired evidence is not that ablation "fixes" hallucinations. A weak or
non-selective causal effect, especially compared with random active features,
supports the representational-marker interpretation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from poc_v10b_logprob_entropy_baseline import load_cache as load_conf_cache
from poc_v9b_token_position_hrm import load_cache as load_position_cache
from src.data.entity_qa import EntityExample, load_jsonl
from src.intervention import compute_gold_logprob, compute_gold_logprob_multi
from src.load import load_all


DATA_PATH = ROOT / "data" / "hotpotqa_v19_300_self_labeled.jsonl"
POSITION_CACHE = ROOT / "outputs" / "cache" / "hotpotqa_v19_300_token_position_sae_features.npz"
CONF_CACHE = ROOT / "outputs" / "cache" / "hotpotqa_v19_300_logprob_entropy_features.npz"
OUT_PATH = ROOT / "outputs" / "poc_v30_hotpotqa_risk_feature_ablation_results.json"

POSITION = "answer_first"
LAYER = 20
SAE_ONLY_C = 0.3
FUSED_C = 0.1
K_VALUES = [2, 4, 8, 12]
N_CORRECT = 30
N_WRONG = 30
N_RANDOM_SEEDS = 3
SAMPLE_SEED = 123
RANDOM_FEATURE_SEED = 1000
ACTIVE_RATE_THRESHOLD = 0.05
SCALE = 1.0


def make_clf(C: float) -> LogisticRegression:
    return LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )


def train_sparse_coef(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    scaler = StandardScaler(with_mean=False)
    X_s = scaler.fit_transform(X.astype(np.float64))
    clf = make_clf(C)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        clf.fit(X_s, y.astype(int))
    return clf.coef_.ravel()


def top_positive_indices(coef: np.ndarray) -> List[int]:
    idx = np.where(coef > 0)[0]
    if len(idx) == 0:
        return []
    order = np.argsort(-coef[idx])
    return idx[order].astype(int).tolist()


def sample_eval_items(items: List[EntityExample], n_correct: int, n_wrong: int, seed: int) -> List[EntityExample]:
    rng = random.Random(seed)
    correct = [x for x in items if x.model_correct is True and x.gold_answers]
    wrong = [x for x in items if x.model_correct is False and x.gold_answers]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    selected = correct[:n_correct] + wrong[:n_wrong]
    return selected


def item_key(item: EntityExample) -> str:
    return f"{item.entity_type}:{item.prompt}"


def mean_std(vals: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def evaluate_feature_set(
    assets,
    items: List[EntityExample],
    baseline_logp: dict[str, float],
    feature_indices: List[int],
    label: str,
) -> dict:
    deltas_correct: List[float] = []
    deltas_wrong: List[float] = []
    token_counts: List[int] = []
    sae = assets.saes[LAYER]
    for pos, item in enumerate(items, start=1):
        gold = (item.gold_answers or [None])[0]
        if not gold:
            continue
        logp_abl, avg_abl, n_tokens = compute_gold_logprob_multi(
            assets.model,
            assets.tokenizer,
            item.prompt,
            gold,
            sae=sae,
            layer=LAYER,
            feature_indices=feature_indices,
            scale=SCALE,
        )
        delta = logp_abl - baseline_logp[item_key(item)]
        token_counts.append(n_tokens)
        if item.model_correct is True:
            deltas_correct.append(delta)
        else:
            deltas_wrong.append(delta)
        if pos % 20 == 0:
            print(f"[v30] {label}: processed {pos}/{len(items)}")

    mc, sc = mean_std(deltas_correct)
    mw, sw = mean_std(deltas_wrong)
    return {
        "label": label,
        "K": len(feature_indices),
        "feature_indices": [int(x) for x in feature_indices],
        "n_correct": len(deltas_correct),
        "n_wrong": len(deltas_wrong),
        "mean_delta_correct": mc,
        "std_delta_correct": sc,
        "mean_delta_wrong": mw,
        "std_delta_wrong": sw,
        "risk_selectivity_wrong_minus_correct": float(mw - mc),
        "mean_answer_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
        "delta_correct": deltas_correct,
        "delta_wrong": deltas_wrong,
    }


def summarize_random(rows: List[dict]) -> dict:
    return {
        "n_runs": len(rows),
        "mean_delta_correct": float(np.mean([r["mean_delta_correct"] for r in rows])),
        "mean_delta_wrong": float(np.mean([r["mean_delta_wrong"] for r in rows])),
        "risk_selectivity_wrong_minus_correct": float(np.mean([r["risk_selectivity_wrong_minus_correct"] for r in rows])),
        "std_selectivity_across_runs": float(np.std([r["risk_selectivity_wrong_minus_correct"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
    }


def main() -> None:
    global LAYER, POSITION, SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="HotpotQA-300")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--position-cache", default=str(POSITION_CACHE))
    parser.add_argument("--conf-cache", default=str(CONF_CACHE))
    parser.add_argument("--position", default=POSITION)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--n-correct", type=int, default=N_CORRECT)
    parser.add_argument("--n-wrong", type=int, default=N_WRONG)
    parser.add_argument("--random-seeds", type=int, default=N_RANDOM_SEEDS)
    parser.add_argument("--sae-only-c", type=float, default=SAE_ONLY_C)
    parser.add_argument("--fused-c", type=float, default=FUSED_C)
    parser.add_argument("--k-values", default=",".join(str(x) for x in K_VALUES))
    parser.add_argument("--scale", type=float, default=SCALE)
    args = parser.parse_args()
    POSITION = args.position
    LAYER = args.layer
    SCALE = args.scale
    data_path = Path(args.data)
    position_cache = Path(args.position_cache)
    conf_cache = Path(args.conf_cache)
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    print("[v30] loading cached features")
    features_by_position, fallback_counts = load_position_cache(position_cache)
    conf_X, conf_labels, conf_names = load_conf_cache(conf_cache)
    sae_X = features_by_position[POSITION][LAYER]["features"].astype(np.float64)
    labels_known = features_by_position[POSITION][LAYER]["labels"].astype(int)
    if not np.array_equal(labels_known, conf_labels.astype(int)):
        raise ValueError("Label/order mismatch between position and confidence caches")
    y_risk = 1 - labels_known

    print("[v30] training sparse probes for feature selection")
    sae_coef = train_sparse_coef(sae_X, y_risk, C=args.sae_only_c)
    sae_top = top_positive_indices(sae_coef)
    fused_X = np.hstack([sae_X, conf_X.astype(np.float64)])
    fused_coef = train_sparse_coef(fused_X, y_risk, C=args.fused_c)
    fused_sae_coef = fused_coef[: sae_X.shape[1]]
    fused_top = top_positive_indices(fused_sae_coef)
    print(f"[v30] SAE-only positive risk features: {len(sae_top)}")
    print(f"[v30] fused positive SAE risk features: {len(fused_top)}")

    activation_rate = (sae_X > 0).mean(axis=0)
    random_pool = np.where(activation_rate > ACTIVE_RATE_THRESHOLD)[0]
    excluded = np.asarray(sorted(set(sae_top + fused_top)), dtype=int)
    if len(excluded):
        random_pool = np.setdiff1d(random_pool, excluded)
    print(f"[v30] random active feature pool: {len(random_pool)}")

    items_all = load_jsonl(str(data_path))
    eval_items = sample_eval_items(items_all, args.n_correct, args.n_wrong, SAMPLE_SEED)
    print(
        f"[v30] eval items: correct={sum(x.model_correct is True for x in eval_items)} "
        f"wrong={sum(x.model_correct is False for x in eval_items)}"
    )

    print("[v30] loading model and SAEs")
    assets = load_all("configs/default.yaml")

    print("[v30] computing baseline gold logprobs once")
    baseline_logp = {}
    baseline_rows = []
    for i, item in enumerate(eval_items, start=1):
        gold = (item.gold_answers or [None])[0]
        logp, avg, n_tokens = compute_gold_logprob(
            assets.model,
            assets.tokenizer,
            item.prompt,
            gold,
            sae=None,
            layer=None,
            feature_idx=None,
            scale=0.0,
        )
        baseline_logp[item_key(item)] = logp
        baseline_rows.append({
            "prompt": item.prompt,
            "gold": gold,
            "model_answer": item.model_answer,
            "model_correct": item.model_correct,
            "baseline_logp": logp,
            "baseline_avg_logp": avg,
            "answer_tokens": n_tokens,
        })
        if i % 20 == 0:
            print(f"[v30] baseline: processed {i}/{len(eval_items)}")

    results = []
    random_results = []
    K_used = [k for k in k_values if k <= len(sae_top) and k <= len(random_pool)]
    print(f"[v30] K values used: {K_used}")
    for K in K_used:
        print(f"[v30] evaluating SAE-only top risk K={K}")
        top_features = sae_top[:K]
        top_row = evaluate_feature_set(
            assets,
            eval_items,
            baseline_logp,
            top_features,
            label=f"sae_only_risk_top_K{K}",
        )
        rand_rows = []
        for seed_i in range(args.random_seeds):
            rng = np.random.default_rng(RANDOM_FEATURE_SEED + seed_i + K * 17)
            rand_features = rng.choice(random_pool, size=K, replace=False).astype(int).tolist()
            print(f"[v30] evaluating random active K={K}, seed={seed_i}, head={rand_features[:5]}")
            rand_rows.append(evaluate_feature_set(
                assets,
                eval_items,
                baseline_logp,
                rand_features,
                label=f"random_active_K{K}_seed{seed_i}",
            ))
        rand_summary = summarize_random(rand_rows)
        top_row["random_avg"] = rand_summary
        top_row["causal_selectivity_vs_random"] = (
            top_row["risk_selectivity_wrong_minus_correct"]
            - rand_summary["risk_selectivity_wrong_minus_correct"]
        )
        results.append(top_row)
        random_results.extend(rand_rows)
        print(
            f"[v30] K={K} top Δcorrect={top_row['mean_delta_correct']:+.4f} "
            f"Δwrong={top_row['mean_delta_wrong']:+.4f} sel={top_row['risk_selectivity_wrong_minus_correct']:+.4f}; "
            f"random sel={rand_summary['risk_selectivity_wrong_minus_correct']:+.4f}"
        )

    fused_results = []
    if fused_top:
        K = min(len(fused_top), len(random_pool))
        print(f"[v30] evaluating fused-positive SAE risk features K={K}")
        fused_results.append(evaluate_feature_set(
            assets,
            eval_items,
            baseline_logp,
            fused_top[:K],
            label=f"fused_positive_sae_risk_K{K}",
        ))

    payload = {
        "experiment": "poc_v30_hotpotqa_risk_feature_ablation",
        "dataset": args.dataset_name,
        "position": POSITION,
        "layer": LAYER,
        "scale": SCALE,
        "feature_selection": {
            "sae_only_C": args.sae_only_c,
            "fused_C": args.fused_c,
            "sae_only_positive_count": len(sae_top),
            "fused_positive_sae_count": len(fused_top),
            "sae_only_top_positive": sae_top[:50],
            "fused_top_positive_sae": fused_top[:50],
        },
        "random_pool": {
            "activation_rate_threshold": ACTIVE_RATE_THRESHOLD,
            "size": int(len(random_pool)),
        },
        "sample": {
            "seed": SAMPLE_SEED,
            "n_correct": sum(x.model_correct is True for x in eval_items),
            "n_wrong": sum(x.model_correct is False for x in eval_items),
        },
        "baseline_rows": baseline_rows,
        "results": results,
        "random_results": random_results,
        "fused_positive_results": fused_results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v30] saved -> {out}")


if __name__ == "__main__":
    main()
