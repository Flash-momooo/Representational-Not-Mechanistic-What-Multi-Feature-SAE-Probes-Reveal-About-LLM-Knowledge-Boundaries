"""V53: Does candidate-conditioned SAE support instability hide a usable subspace?"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load import load_config, load_saes  # noqa: E402

SEED = 20260828
N_REPLICATES = 12
SUBSAMPLE_FRACTION = 0.60
K_VALUES = (8, 32)
BATCH_SIZE = 128
N_BOOTSTRAP = 10_000

SOURCE_CACHE = ROOT / "outputs/cache/equal_compute_source.npz"
TARGET_RAW_CACHE = ROOT / "outputs/cache/v48_hotpot_scale_candidate_readout.npz"
TARGET_SAE_CACHE = ROOT / "outputs/cache/v53_hotpot_candidate_gemmascope_sae.npz"
RESULT = ROOT / "outputs/poc_v53_candidate_conditioned_subspace_audit.json"
ARTIFACT = ROOT / "outputs/cache/poc_v53_candidate_conditioned_subspace_artifacts.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def centre(values: np.ndarray, question_ids: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for question_id in np.unique(question_ids.astype(str)):
        mask = question_ids.astype(str) == question_id
        block = values[mask].astype(np.float32)
        result[mask] = block - block.mean(axis=0, keepdims=True)
    return result


def encode_target_sae(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if TARGET_SAE_CACHE.exists():
        cached = load_npz(TARGET_SAE_CACHE)
        return cached["sae"].astype(np.float32), cached["decoder"].astype(np.float32)

    cfg = load_config(str(ROOT / "configs/fpe5_l18.yaml"))
    cfg["sae"]["layers"] = [18]
    sae = load_saes(cfg)[18]
    parts: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(raw), BATCH_SIZE):
                batch = torch.from_numpy(raw[start:start + BATCH_SIZE]).to(
                    device=cfg["model"]["device"], dtype=getattr(torch, cfg["model"]["dtype"])
                )
                parts.append(sae.encode(batch).float().cpu().numpy().astype(np.float16))
        decoder_tensor = getattr(sae, "W_dec", None)
        if decoder_tensor is None:
            decoder_tensor = getattr(sae, "decoder", None)
        if decoder_tensor is None:
            raise RuntimeError("The loaded SAE exposes neither W_dec nor decoder")
        decoder = decoder_tensor.detach().float().cpu().numpy().astype(np.float32)
    finally:
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TARGET_SAE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TARGET_SAE_CACHE, sae=np.concatenate(parts), decoder=decoder)
    return np.concatenate(parts).astype(np.float32), decoder


def active_pool(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.mean(values[train] > 0.0, axis=0) >= 0.01).astype(np.int32)


def standardised_difference(values: np.ndarray, labels: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = values[train]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.maximum(std, 1e-6)
    standardised = (values - mean) / std
    positives = standardised[train & (labels == 1)]
    negatives = standardised[train & (labels == 0)]
    weights = positives.mean(axis=0) - negatives.mean(axis=0)
    return weights.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def score(values: np.ndarray, mean: np.ndarray, std: np.ndarray, support: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (((values[:, support] - mean[support]) / std[support]) @ weights[support]).astype(np.float32)


def question_metrics(labels: np.ndarray, scores: np.ndarray, question_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aucs, selected = [], []
    for question_id in np.unique(question_ids.astype(str)):
        mask = question_ids.astype(str) == question_id
        aucs.append(float(roc_auc_score(labels[mask], scores[mask])))
        selected.append(float(labels[mask][np.argmin(scores[mask])] == 0))
    return np.asarray(aucs, dtype=np.float64), np.asarray(selected, dtype=np.float64)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom else float("nan")


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    lhs, rhs = set(map(int, left)), set(map(int, right))
    return len(lhs & rhs) / max(len(lhs | rhs), 1)


def span_overlap(decoder: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    left_basis, _ = np.linalg.qr(decoder[left].T, mode="reduced")
    right_basis, _ = np.linalg.qr(decoder[right].T, mode="reduced")
    singular_values = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    return float(np.mean(np.square(np.clip(singular_values, 0.0, 1.0))))


def summary(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "min": float(array.min()), "max": float(array.max()), "n": int(len(array)),
    }


def bootstrap_pair(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "n_questions": int(len(values)),
    }


def aggregate(records: list[dict], kind: str, decoder: np.ndarray, target_y: np.ndarray, target_qids: np.ndarray) -> dict:
    pairs = list(combinations(range(len(records)), 2))
    supports = [item[f"{kind}_support"] for item in records]
    directions = [item[f"{kind}_direction"] for item in records]
    scores = [item[f"{kind}_score"] for item in records]
    aucs, selections = zip(*(question_metrics(target_y, value, target_qids) for value in scores))
    return {
        "feature_jaccard": summary([jaccard(supports[i], supports[j]) for i, j in pairs]),
        "decoder_span_overlap": summary([span_overlap(decoder, supports[i], supports[j]) for i, j in pairs]),
        "residual_direction_cosine": summary([cosine(directions[i], directions[j]) for i, j in pairs]),
        "target_score_spearman": summary([spearmanr(scores[i], scores[j]).statistic for i, j in pairs]),
        "target_within_question_auroc": summary([value.mean() for value in aucs]),
        "target_selection_accuracy": summary([value.mean() for value in selections]),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("V53 loading source and frozen target states", flush=True)
    source, target = load_npz(SOURCE_CACHE), load_npz(TARGET_RAW_CACHE)
    target_sae, decoder = encode_target_sae(target["raw"].astype(np.float32))

    source_values = centre(source["sae"].astype(np.float32), source["question_ids"])
    source_y = source["original_risk"].astype(np.int8)
    source_qids = source["question_ids"].astype(str)
    target_values = centre(target_sae, target["question_ids"])
    target_y = target["risk"].astype(np.int8)
    target_qids = target["question_ids"].astype(str)
    questions = np.unique(source_qids)
    target_questions = np.unique(target_qids)
    if len(target_questions) != 500 or not np.all(np.unique(np.unique(target_qids, return_counts=True)[1]) == 4):
        raise RuntimeError("Expected four V47 candidates per target question")
    if not np.array_equal(np.sort(np.unique(source_y)), np.asarray([0, 1])):
        raise RuntimeError("Source needs both correct and incorrect candidates")

    records_by_k: dict[int, list[dict]] = {k: [] for k in K_VALUES}
    artifact: dict[str, np.ndarray] = {"source_questions": questions, "target_questions": target_questions}
    for repeat in range(N_REPLICATES):
        local_rng = np.random.default_rng(SEED + repeat)
        sampled = local_rng.choice(questions, size=int(round(len(questions) * SUBSAMPLE_FRACTION)), replace=False)
        train = np.isin(source_qids, sampled)
        weights, mean, std = standardised_difference(source_values, source_y, train)
        pool = active_pool(source_values, train)
        if len(pool) < max(K_VALUES):
            raise RuntimeError(f"Only {len(pool)} active SAE features; cannot use requested K")
        ranked = np.argsort(-np.abs(weights)).astype(np.int32)
        permuted_y = source_y.copy()
        for question_id in sampled:
            mask = np.flatnonzero((source_qids == question_id) & train)
            permuted_y[mask] = local_rng.permutation(permuted_y[mask])
        perm_weights, perm_mean, perm_std = standardised_difference(source_values, permuted_y, train)
        for k in K_VALUES:
            selected = np.sort(ranked[:k])
            random_support = np.sort(local_rng.choice(pool, size=k, replace=False)).astype(np.int32)
            permuted = np.sort(np.argsort(-np.abs(perm_weights))[:k]).astype(np.int32)
            selected_score = score(target_values, mean, std, selected, weights)
            random_score = score(target_values, mean, std, random_support, weights)
            permuted_score = score(target_values, perm_mean, perm_std, permuted, perm_weights)
            record = {
                "selected_support": selected,
                "random_support": random_support,
                "permuted_support": permuted,
                "selected_direction": weights[selected] @ decoder[selected],
                "random_direction": weights[random_support] @ decoder[random_support],
                "permuted_direction": perm_weights[permuted] @ decoder[permuted],
                "selected_score": selected_score,
                "random_score": random_score,
                "permuted_score": permuted_score,
            }
            records_by_k[k].append(record)

    results: dict[str, dict] = {}
    full_source_evaluation: dict[str, dict] = {}
    for k, records in records_by_k.items():
        selected = aggregate(records, "selected", decoder, target_y, target_qids)
        random = aggregate(records, "random", decoder, target_y, target_qids)
        permuted = aggregate(records, "permuted", decoder, target_y, target_qids)
        results[f"K{k}"] = {
            "selected_support": selected,
            "active_random_control": random,
            "within_question_label_permutation_control": permuted,
            "selected_minus_random": {
                key: float(selected[key]["mean"] - random[key]["mean"])
                for key in ("decoder_span_overlap", "residual_direction_cosine", "target_score_spearman", "target_within_question_auroc", "target_selection_accuracy")
            },
        }
        for kind in ("selected", "random", "permuted"):
            artifact[f"K{k}_{kind}_support"] = np.stack([record[f"{kind}_support"] for record in records])
            artifact[f"K{k}_{kind}_direction"] = np.stack([record[f"{kind}_direction"] for record in records])
            artifact[f"K{k}_{kind}_score"] = np.stack([record[f"{kind}_score"] for record in records])

        # This is a single source-only fit followed by a question bootstrap on
        # V47. It is kept separate from subsample stability, which estimates
        # reproducibility rather than a target uncertainty interval.
        full_train = np.ones(len(source_y), dtype=bool)
        full_weights, full_mean, full_std = standardised_difference(source_values, source_y, full_train)
        full_pool = active_pool(source_values, full_train)
        full_selected = np.sort(np.argsort(-np.abs(full_weights))[:k]).astype(np.int32)
        full_rng = np.random.default_rng(SEED + 10000 + k)
        full_random = np.sort(full_rng.choice(full_pool, size=k, replace=False)).astype(np.int32)
        full_selected_score = score(target_values, full_mean, full_std, full_selected, full_weights)
        full_random_score = score(target_values, full_mean, full_std, full_random, full_weights)
        selected_auc, selected_accuracy = question_metrics(target_y, full_selected_score, target_qids)
        random_auc, random_accuracy = question_metrics(target_y, full_random_score, target_qids)
        full_source_evaluation[f"K{k}"] = {
            "source_only_selected_support": full_selected.tolist(),
            "source_only_random_support": full_random.tolist(),
            "selected": {
                "within_question_auroc": bootstrap_pair(selected_auc, SEED + k),
                "selection_accuracy": bootstrap_pair(selected_accuracy, SEED + k + 1),
            },
            "active_random_control": {
                "within_question_auroc": bootstrap_pair(random_auc, SEED + k + 2),
                "selection_accuracy": bootstrap_pair(random_accuracy, SEED + k + 3),
            },
            "paired_selected_minus_random": {
                "within_question_auroc": bootstrap_pair(selected_auc - random_auc, SEED + k + 4),
                "selection_accuracy": bootstrap_pair(selected_accuracy - random_accuracy, SEED + k + 5),
            },
        }
        artifact[f"K{k}_full_selected_support"] = full_selected
        artifact[f"K{k}_full_random_support"] = full_random
        artifact[f"K{k}_full_selected_score"] = full_selected_score
        artifact[f"K{k}_full_random_score"] = full_random_score

    payload = {
        "experiment": "V53 candidate-conditioned SAE subspace audit",
        "status": "completed",
        "protocol": "paper/V53_CANDIDATE_CONDITIONED_SUBSPACE_AUDIT_PROTOCOL.md",
        "scope": "Post-hoc diagnostic audit on the V47/V48 target; target states are never used in support selection.",
        "source": {"questions": int(len(questions)), "candidate_states": int(len(source_y)), "subsample_fraction": SUBSAMPLE_FRACTION, "n_grouped_subsamples": N_REPLICATES},
        "target": {"questions": int(len(target_questions)), "candidate_states": int(len(target_y)), "candidates_per_question": 4, "source_id_isolated_at_V47_construction": True},
        "representation": {"sae": "GemmaScope layer-18 width-16k", "decoder_shape": list(decoder.shape), "question_centered": True, "k_values": list(K_VALUES)},
        "pre_specified_interpretation": "Stable usable geometry requires selected supports to exceed active-random controls on decoder geometry, target score agreement, and target candidate discrimination. Low Jaccard alone is not a mechanism claim.",
        "results": results,
        "single_source_fit_target_bootstrap": full_source_evaluation,
        "limitations": ["This does not establish a causal circuit or a cross-model invariant subspace.", "This is a post-hoc V47 diagnostic audit, not a second prospective confirmation.", "Candidate correctness is operationalized by the supplied machine-verifiable option labels."],
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(ARTIFACT, **artifact)
    print(f"saved {RESULT}", flush=True)
    for key, item in results.items():
        value = item["selected_support"]
        control = item["active_random_control"]
        print(
            f"{key}: J={value['feature_jaccard']['mean']:.3f}; "
            f"span={value['decoder_span_overlap']['mean']:.3f}/{control['decoder_span_overlap']['mean']:.3f}; "
            f"dir={value['residual_direction_cosine']['mean']:.3f}/{control['residual_direction_cosine']['mean']:.3f}; "
            f"rank={value['target_score_spearman']['mean']:.3f}/{control['target_score_spearman']['mean']:.3f}; "
            f"AUC={value['target_within_question_auroc']['mean']:.3f}/{control['target_within_question_auroc']['mean']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
