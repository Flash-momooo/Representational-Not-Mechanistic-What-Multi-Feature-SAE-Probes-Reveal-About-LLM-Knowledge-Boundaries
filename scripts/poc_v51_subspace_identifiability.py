"""V51: audit whether unstable SAE supports conceal a stable predictive subspace.

This is an analysis on the fixed Qwen-7B true-SAE WebQuestions cache.  A fixed
question-level holdout is never used to select features.  Repeated grouped
development subsamples independently select SAE supports.  We then compare:

1. feature-ID Jaccard overlap;
2. overlap of the corresponding SAE decoder row spans;
3. agreement of the signed decoder-space risk directions; and
4. retained heldout prediction under selected versus random supports.

The experiment deliberately distinguishes a stable *subspace* from a stable
feature identity.  It does not turn either result into a causal claim.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_frozen_representation_controls import load_condition, raw_and_confidence
from scripts.poc_fpe3_trajectory_dynamics import fit_scalers, transform
from scripts.poc_fpe6_single_trajectory_filter import fit_logistic, logistic_predict, solve_ridge
from scripts.poc_fpe8_sparse_risk_distillation import fit_sparse_student, predict_sparse_student
from scripts.poc_nn32_train_qwen7b_true_sae import TopKSAE


DATA = ROOT / "data" / "nn19_qwen7b_webquestions_trajectory.jsonl"
CACHE = ROOT / "outputs" / "cache" / "nn32_qwen7b_webquestions_true_sae_states.npz"
WEIGHTS = ROOT / "outputs" / "nn32_qwen7b_l18_topk_sae.pt"
TOKENIZER = ROOT / "models" / "Qwen2.5-7B-Instruct"
RESULT = ROOT / "outputs" / "poc_v51_subspace_identifiability_results.json"
ARTIFACT = ROOT / "outputs" / "cache" / "poc_v51_subspace_identifiability_artifacts.npz"

SEED = 20260828
STAGES = (1, 2, 3)
# K=8 is deliberately used for the primary audit.  At K=32 the active-feature
# pool at some early stages is nearly exhausted, making equal-K random supports
# overlap by construction rather than providing a meaningful geometry control.
K_VALUES = (8,)
N_REPLICATES = 12
HOLDOUT_FRACTION = 0.20
DEV_SUBSAMPLE_FRACTION = 0.60
MIN_ACTIVE_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set, right_set = set(map(int, left)), set(map(int, right))
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def span_overlap(decoder: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Mean squared cosine of principal angles between two decoder row spans."""
    left_basis, _ = np.linalg.qr(decoder[left].T, mode="reduced")
    right_basis, _ = np.linalg.qr(decoder[right].T, mode="reduced")
    singular_values = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    return float(np.mean(np.square(np.clip(singular_values, 0.0, 1.0))))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def summary(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def conditional_ridge(
    values: np.ndarray, confidence: np.ndarray, labels: np.ndarray, train: np.ndarray
) -> dict:
    """Fit the same confidence-residualized ridge used for sparse support ranking."""
    confidence_scaler = fit_scalers([confidence], train)
    confidence_features = transform([confidence], confidence_scaler)
    baseline = fit_logistic(confidence_features[train], labels[train], ridge=0.05)
    residual_target = labels.astype(np.float64) - logistic_predict(baseline, confidence_features)
    value_scaler = fit_scalers([values], train)
    value_features = transform([values], value_scaler)
    _, weights = solve_ridge(value_features[train], residual_target[train], DEVICE)
    return {
        "weights": weights.astype(np.float32),
        "value_scaler": value_scaler,
        "value_features": value_features,
    }


def residual_direction(model: dict, support: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    """Map standardized SAE coefficients back into the residual decoder space."""
    _, rms = model["value_scaler"][0]
    d = model["value_features"].shape[1]
    original_scale = rms.reshape(-1) * np.sqrt(d)
    coefficients = model["weights"][support] / original_scale[support]
    return coefficients.astype(np.float32) @ decoder[support]


def evaluate_support(
    values: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    support: np.ndarray,
) -> tuple[float, np.ndarray]:
    model = fit_sparse_student(values, confidence, labels, labels, train, support)
    probability = predict_sparse_student(model, values, confidence)
    return float(roc_auc_score(labels[test], probability[test])), probability[test].astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print(f"V51 device={DEVICE}; loading fixed cache", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    states, table = load_condition(tokenizer, DATA, CACHE)
    labels_final = states["labels"].astype(np.int8)
    question_ids = states["question_ids"].astype(str)

    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sae = TopKSAE(config["d_model"], config["d_sae"], config["top_k"], checkpoint["state_dict"]["mean"])
    sae.load_state_dict(checkpoint["state_dict"])
    decoder = sae.decoder.detach().cpu().numpy().astype(np.float32)
    del sae

    questions = np.unique(question_ids)
    shuffled_questions = rng.permutation(questions)
    n_holdout = int(round(HOLDOUT_FRACTION * len(questions)))
    holdout_questions = shuffled_questions[:n_holdout]
    development_questions = shuffled_questions[n_holdout:]
    fixed_holdout = np.isin(question_ids, holdout_questions)
    print(
        f"questions: development={len(development_questions)}, heldout={len(holdout_questions)}; "
        f"replicates={N_REPLICATES}",
        flush=True,
    )

    records: dict[str, dict] = {}
    artifact: dict[str, np.ndarray] = {"heldout_questions": holdout_questions.astype(str)}
    for stage in STAGES:
        values, confidence = raw_and_confidence(states, stage, "sae")
        valid = table["risk"][stage].astype(bool)
        test = fixed_holdout & valid
        if len(np.unique(labels_final[test])) < 2:
            raise RuntimeError(f"T{stage} heldout has only one event class")
        active_rate = np.mean(values[(~fixed_holdout) & valid] > 0, axis=0)
        active_pool = np.flatnonzero(active_rate >= MIN_ACTIVE_RATE).astype(np.int32)
        if len(active_pool) < max(K_VALUES):
            raise RuntimeError(f"T{stage} has too few active SAE features")
        per_rep = []
        print(f"T{stage}: active_pool={len(active_pool)}, fitting grouped subsamples", flush=True)
        for replicate in range(N_REPLICATES):
            local_rng = np.random.default_rng(SEED + stage * 1000 + replicate)
            selected_questions = local_rng.choice(
                development_questions,
                size=int(round(DEV_SUBSAMPLE_FRACTION * len(development_questions))),
                replace=False,
            )
            train = np.isin(question_ids, selected_questions) & valid
            if len(np.unique(labels_final[train])) < 2:
                raise RuntimeError(f"T{stage}, replicate {replicate} has only one event class")
            full = conditional_ridge(values, confidence, labels_final, train)
            support_order = np.argsort(-np.abs(full["weights"]))
            item: dict[str, object] = {"replicate": replicate, "n_train_rows": int(train.sum())}
            for k in K_VALUES:
                support = np.sort(support_order[:k]).astype(np.int32)
                random_support = np.sort(local_rng.choice(active_pool, size=k, replace=False)).astype(np.int32)
                random_model = conditional_ridge(values[:, random_support], confidence, labels_final, train)
                selected_direction = residual_direction(full, support, decoder)
                random_direction_small = residual_direction(random_model, np.arange(k, dtype=np.int32), decoder[random_support])
                selected_auc, selected_probability = evaluate_support(values, confidence, labels_final, train, test, support)
                random_auc, random_probability = evaluate_support(values, confidence, labels_final, train, test, random_support)
                # This score excludes confidence entirely.  It is the direct
                # heldout test of the selected SAE direction rather than of a
                # confidence-plus-SAE sparse student.
                selected_residual_score = full["value_features"][:, support] @ full["weights"][support]
                random_residual_score = random_model["value_features"] @ random_model["weights"]
                selected_residual_auc = float(roc_auc_score(labels_final[test], selected_residual_score[test]))
                random_residual_auc = float(roc_auc_score(labels_final[test], random_residual_score[test]))
                item[str(k)] = {
                    "support": support.tolist(),
                    "random_support": random_support.tolist(),
                    "selected_direction": selected_direction.astype(np.float32),
                    "random_direction": random_direction_small.astype(np.float32),
                    "selected_auc": selected_auc,
                    "random_auc": random_auc,
                    "selected_residual_auc": selected_residual_auc,
                    "random_residual_auc": random_residual_auc,
                    "selected_probability": selected_probability,
                    "random_probability": random_probability,
                }
            per_rep.append(item)

        for k in K_VALUES:
            selected_sets = [np.asarray(row[str(k)]["support"], dtype=np.int32) for row in per_rep]
            random_sets = [np.asarray(row[str(k)]["random_support"], dtype=np.int32) for row in per_rep]
            selected_directions = [np.asarray(row[str(k)]["selected_direction"], dtype=np.float32) for row in per_rep]
            random_directions = [np.asarray(row[str(k)]["random_direction"], dtype=np.float32) for row in per_rep]
            selected_probs = [np.asarray(row[str(k)]["selected_probability"], dtype=np.float32) for row in per_rep]
            random_probs = [np.asarray(row[str(k)]["random_probability"], dtype=np.float32) for row in per_rep]
            pairs = list(combinations(range(N_REPLICATES), 2))
            selected_jaccard = [jaccard(selected_sets[i], selected_sets[j]) for i, j in pairs]
            random_jaccard = [jaccard(random_sets[i], random_sets[j]) for i, j in pairs]
            selected_span = [span_overlap(decoder, selected_sets[i], selected_sets[j]) for i, j in pairs]
            random_span = [span_overlap(decoder, random_sets[i], random_sets[j]) for i, j in pairs]
            selected_cosine = [cosine(selected_directions[i], selected_directions[j]) for i, j in pairs]
            random_cosine = [cosine(random_directions[i], random_directions[j]) for i, j in pairs]
            selected_spearman = [spearmanr(selected_probs[i], selected_probs[j]).statistic for i, j in pairs]
            random_spearman = [spearmanr(random_probs[i], random_probs[j]).statistic for i, j in pairs]
            selected_aucs = [float(row[str(k)]["selected_auc"]) for row in per_rep]
            random_aucs = [float(row[str(k)]["random_auc"]) for row in per_rep]
            selected_residual_aucs = [float(row[str(k)]["selected_residual_auc"]) for row in per_rep]
            random_residual_aucs = [float(row[str(k)]["random_residual_auc"]) for row in per_rep]
            key = f"T{stage}_K{k}"
            records[key] = {
                "stage": stage,
                "k": k,
                "heldout_rows": int(test.sum()),
                "heldout_event_rate": float(labels_final[test].mean()),
                "selected_support_jaccard": summary(selected_jaccard),
                "random_support_jaccard": summary(random_jaccard),
                "selected_decoder_span_overlap": summary(selected_span),
                "random_decoder_span_overlap": summary(random_span),
                "span_overlap_advantage": float(np.mean(selected_span) - np.mean(random_span)),
                "selected_decoder_direction_cosine": summary(selected_cosine),
                "random_decoder_direction_cosine": summary(random_cosine),
                "direction_cosine_advantage": float(np.nanmean(selected_cosine) - np.nanmean(random_cosine)),
                "selected_heldout_auc": summary(selected_aucs),
                "random_heldout_auc": summary(random_aucs),
                "heldout_auc_advantage": float(np.mean(selected_aucs) - np.mean(random_aucs)),
                "selected_SAE_only_heldout_auc": summary(selected_residual_aucs),
                "random_SAE_only_heldout_auc": summary(random_residual_aucs),
                "SAE_only_heldout_auc_advantage": float(np.mean(selected_residual_aucs) - np.mean(random_residual_aucs)),
                "selected_probability_spearman": summary(selected_spearman),
                "random_probability_spearman": summary(random_spearman),
            }
            artifact[f"{key}_support"] = np.stack(selected_sets)
            artifact[f"{key}_random_support"] = np.stack(random_sets)
            artifact[f"{key}_direction"] = np.stack(selected_directions)
            artifact[f"{key}_random_direction"] = np.stack(random_directions)
        del per_rep

    payload = {
        "experiment": "V51 grouped-subsample SAE subspace-identifiability audit",
        "status": "completed",
        "scope": "Fixed Qwen2.5-7B-Instruct true Top-K SAE and WebQuestions trajectory cache; this is not a cross-model result.",
        "pre_specified_interpretation": (
            "A stable-subspace finding requires decoder-span overlap and signed decoder-direction agreement "
            "above equal-K random supports while preserving heldout discrimination. Coordinate Jaccard alone is insufficient."
        ),
        "data_split": {
            "seed": SEED,
            "n_questions": int(len(questions)),
            "development_questions": int(len(development_questions)),
            "heldout_questions": int(len(holdout_questions)),
            "development_subsample_fraction": DEV_SUBSAMPLE_FRACTION,
            "n_grouped_subsamples": N_REPLICATES,
            "heldout_question_ids_saved_in": str(ARTIFACT.relative_to(ROOT)),
        },
        "representation": {
            "dictionary": str(WEIGHTS.relative_to(ROOT)),
            "decoder_shape": list(decoder.shape),
            "stages": list(STAGES),
            "k_values": list(K_VALUES),
            "random_reference": "same-K features sampled from features active in at least 1% of development rows",
        },
        "results": records,
        "limitations": (
            "Decoder-span agreement is representational geometry, not a causal circuit test. "
            "The fixed heldout is used repeatedly for reporting across resamples and should be treated as an audit set, not a new prospective dataset."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(ARTIFACT, **artifact)
    print(f"saved -> {RESULT}", flush=True)
    for key, row in records.items():
        print(
            f"{key}: J={row['selected_support_jaccard']['mean']:.3f}; "
            f"span={row['selected_decoder_span_overlap']['mean']:.3f} vs random {row['random_decoder_span_overlap']['mean']:.3f}; "
            f"dir={row['selected_decoder_direction_cosine']['mean']:.3f} vs random {row['random_decoder_direction_cosine']['mean']:.3f}; "
            f"SAE-only AUC={row['selected_SAE_only_heldout_auc']['mean']:.3f} vs random {row['random_SAE_only_heldout_auc']['mean']:.3f}; "
            f"student AUC={row['selected_heldout_auc']['mean']:.3f} vs random {row['random_heldout_auc']['mean']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
