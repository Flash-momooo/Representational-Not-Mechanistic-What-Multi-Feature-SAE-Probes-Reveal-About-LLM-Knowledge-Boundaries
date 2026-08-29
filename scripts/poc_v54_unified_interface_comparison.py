"""V54: compare monitoring interfaces on one fixed candidate-conditioned task."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import load_npz  # noqa: E402
from scripts.poc_equal_compute_commitment_factorial import (  # noqa: E402
    condition_sequences,
)
from scripts.poc_v40_extract_and_evaluate import padded_batch  # noqa: E402
from scripts.poc_v48_scale_candidate_readout import centre, load_items, source_model  # noqa: E402
from src.load import load_config, load_model_and_tokenizer  # noqa: E402

SEED = 20260828
N_BOOTSTRAP = 10_000
PCA_RANKS = (8, 16, 32, 64)
LAYER = 18


def bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "n_questions": int(len(values)),
    }


def question_metrics(risk: np.ndarray, score: np.ndarray, qids: np.ndarray, seed: int) -> dict:
    aucs: list[float] = []
    selected: list[float] = []
    for qid in np.unique(qids):
        indices = np.flatnonzero(qids == qid)
        aucs.append(float(roc_auc_score(risk[indices], score[indices])))
        selected.append(float(risk[indices[np.argmin(score[indices])]] == 0))
    auc_array = np.asarray(aucs, dtype=np.float64)
    selected_array = np.asarray(selected, dtype=np.float64)
    return {
        "within_question_auroc": bootstrap(auc_array, seed),
        "selection_accuracy": bootstrap(selected_array, seed + 1),
        "population_auroc": float(roc_auc_score(risk, score)),
        "population_auprc": float(average_precision_score(risk, score)),
        "per_question_auroc": auc_array,
        "per_question_selection": selected_array,
    }


def source_real(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    condition = source.get("condition")
    if condition is None:
        return source
    keep = condition.astype(str) == "real"
    return {key: value[keep] for key, value in source.items() if value.shape[0] == len(keep)}


def fit_scalar_risk(source_feature: np.ndarray, source_risk: np.ndarray, target_feature: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=SEED),
    )
    model.fit(source_feature.reshape(-1, 1), source_risk)
    return model.predict_proba(target_feature.reshape(-1, 1))[:, 1]


def pca_cv_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray, components: int) -> float:
    folds = GroupKFold(n_splits=4)
    scores: list[float] = []
    for train, valid in folds.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, svd_solver="randomized", random_state=SEED),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=SEED),
        )
        model.fit(x[train], y[train])
        scores.append(float(roc_auc_score(y[valid], model.predict_proba(x[valid])[:, 1])))
    return float(np.mean(scores))


def fit_pca_risk(source_x: np.ndarray, source_y: np.ndarray, source_qids: np.ndarray, target_x: np.ndarray) -> tuple[np.ndarray, dict]:
    cv = {str(rank): pca_cv_score(source_x, source_y, source_qids, rank) for rank in PCA_RANKS}
    selected = min(PCA_RANKS, key=lambda rank: (-cv[str(rank)], rank))
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=selected, svd_solver="randomized", random_state=SEED),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=SEED),
    )
    model.fit(source_x, source_y)
    return model.predict_proba(target_x)[:, 1], {"candidate_ranks": list(PCA_RANKS), "source_grouped_cv_population_auroc": cv, "selected_rank": selected}


def extract_target_local_stats(items: list[dict], config: str) -> dict[str, np.ndarray]:
    """Extract lexical uncertainty from exactly the existing candidate prompts."""
    cfg = load_config(config)
    model, tokenizer = load_model_and_tokenizer(cfg)
    rows = {"output_entropy": [], "input_tokens": []}
    try:
        for item in tqdm(items, desc="v54-local-uncertainty"):
            sequences, _, _ = condition_sequences(tokenizer, item, "real")
            ids, mask, lengths = padded_batch(sequences, int(tokenizer.eos_token_id), cfg["model"]["device"])
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
            batch = torch.arange(len(sequences), device=ids.device)
            # The candidate is complete at `last`. Reading logits at `prior`
            # would test a shared state before the final candidate token enters
            # the context, not a candidate-conditioned uncertainty interface.
            last = torch.tensor(lengths - 1, device=ids.device)
            log_probs = torch.log_softmax(output.logits[batch, last].float(), dim=-1)
            probabilities = log_probs.exp()
            rows["output_entropy"].append((-(probabilities * log_probs).sum(dim=-1)).cpu().numpy().astype(np.float32))
            rows["input_tokens"].append(np.asarray(lengths, dtype=np.int32))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {key: np.concatenate(values) for key, values in rows.items()}


def load_or_extract_stats(items_path: Path, cache_path: Path, config: str) -> dict[str, np.ndarray]:
    if cache_path.exists():
        return load_npz(cache_path)
    stats = extract_target_local_stats(load_items(items_path), config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **stats)
    return stats


def strip_arrays(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if not key.startswith("per_question_")}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", default="outputs/cache/equal_compute_source.npz")
    parser.add_argument("--target-cache", default="outputs/cache/v48_hotpot_scale_candidate_readout.npz")
    parser.add_argument("--sae-artifact", default="outputs/cache/poc_v53_candidate_conditioned_subspace_artifacts.npz")
    parser.add_argument("--items", default="data/v47_hotpot_scale_confirmation_candidates.jsonl")
    parser.add_argument("--local-stats-cache", default="outputs/cache/v54_hotpot_postcandidate_uncertainty.npz")
    parser.add_argument("--config", default="configs/fpe5_l18.yaml")
    parser.add_argument("--external", default="outputs/poc_v49_zero_shot_qwen7b_candidate_verifier.json")
    parser.add_argument("--output", default="outputs/poc_v54_unified_interface_comparison.json")
    args = parser.parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    source = source_real(load_npz(ROOT / args.source_cache))
    target = load_npz(ROOT / args.target_cache)
    artifact = load_npz(ROOT / args.sae_artifact)
    local = load_or_extract_stats(ROOT / args.items, ROOT / args.local_stats_cache, args.config)
    qids = target["question_ids"].astype(str)
    risk = target["risk"].astype(np.int8)
    source_qids = source["question_ids"].astype(str)
    source_risk = source["original_risk"].astype(np.int8)
    if len(np.unique(qids)) != 500 or not np.all(np.unique(np.unique(qids, return_counts=True)[1]) == 4):
        raise RuntimeError("Expected 500 target questions with four candidates each")
    if len(local["output_entropy"]) != len(risk):
        raise RuntimeError("Local uncertainty cache does not align with the V48 target")

    source_raw = centre(source["raw"].astype(np.float32), source_qids)
    target_raw = centre(target["raw"].astype(np.float32), qids)
    dense_scaler, dense_clf = source_model({
        "condition": np.asarray(["real"] * len(source_risk), dtype=object),
        "raw": source["raw"],
        "question_ids": source_qids,
        "original_risk": source_risk,
    })
    scores: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    scores["dense_residual"] = dense_clf.predict_proba(dense_scaler.transform(target_raw))[:, 1]
    metadata["dense_residual"] = {"interface": "source-fit logistic readout of question-centred layer-18 residual", "same_gemma_forward": True}

    scores["sae_top32"] = artifact["K32_full_selected_score"].astype(np.float64)
    metadata["sae_top32"] = {"interface": "source-only Top-32 GemmaScope SAE support from V53", "same_gemma_forward": True}

    pca_score, pca_meta = fit_pca_risk(source_raw, source_risk, source_qids, target_raw)
    scores["pca_readout"] = pca_score
    metadata["pca_readout"] = {"interface": "source-selected PCA plus logistic readout", "same_gemma_forward": True, **pca_meta}

    scores["candidate_nll"] = -target["label_logprob"].astype(np.float64)
    metadata["candidate_nll"] = {"interface": "negative final-candidate-token log probability", "same_gemma_forward": True}

    scores["next_token_entropy"] = local["output_entropy"].astype(np.float64)
    metadata["next_token_entropy"] = {"interface": "entropy of the next-token distribution after the complete candidate", "same_gemma_forward": True, "not_semantic_entropy": True}

    source_norm = np.linalg.norm(source_raw, axis=1)
    target_norm = np.linalg.norm(target_raw, axis=1)
    scores["residual_norm"] = fit_scalar_risk(source_norm, source_risk, target_norm)
    metadata["residual_norm"] = {"interface": "source-fit scalar readout of question-centred residual norm", "same_gemma_forward": True}

    metrics = {name: question_metrics(risk, score, qids, SEED + 17 * index) for index, (name, score) in enumerate(scores.items())}
    dense_metrics = metrics["dense_residual"]
    comparisons = {}
    for index, (name, record) in enumerate(metrics.items()):
        if name == "dense_residual":
            continue
        comparisons[f"dense_minus_{name}"] = {
            "within_question_auroc": bootstrap(dense_metrics["per_question_auroc"] - record["per_question_auroc"], SEED + 200 + index),
            "selection_accuracy": bootstrap(dense_metrics["per_question_selection"] - record["per_question_selection"], SEED + 300 + index),
        }

    external_payload = json.loads((ROOT / args.external).read_text(encoding="utf-8"))
    external = next(iter(external_payload["methods"].values()))["target"]
    payload = {
        "experiment": "V54 unified candidate-conditioned interface comparison",
        "protocol": "paper/V54_UNIFIED_INTERFACE_COMPARISON_PROTOCOL.md",
        "status": "post-hoc interface audit on the previously examined V47/V48 target; all target-side scores are evaluated once without target refitting",
        "target": {"questions": int(len(np.unique(qids))), "candidate_states": int(len(qids)), "candidates_per_question": 4},
        "source": {"questions": int(len(np.unique(source_qids))), "candidate_states": int(len(source_qids))},
        "shared_compute": {
            "backbone": "Gemma-2-2B layer 18",
            "candidate_conditioned_forwards": int(len(qids)),
            "additional_stochastic_samples": 0,
            "mean_gemma_input_tokens_per_candidate": float(local["input_tokens"].mean()),
            "note": "Dense, SAE, PCA, NLL, entropy, and norm use the same candidate-conditioned Gemma forward pass."
        },
        "internal_interfaces": {name: {**metadata[name], **strip_arrays(record)} for name, record in metrics.items()},
        "paired_dense_minus_internal": comparisons,
        "external_reference_not_same_backbone": {
            "name": "zero-shot Qwen2.5-7B supported-versus-unsupported verifier",
            "metrics": {key: value for key, value in external.items() if key != "selected_indices"},
            "reason_separate": "different model, prompt, and resource regime; included as a task-aligned external-verification reference only",
        },
        "not_evaluated_under_this_contract": {
            "semantic_entropy": "requires independently sampled completions; fixed answer options are not samples",
            "SelfCheckGPT": "requires multiple stochastic textual generations; fixed answer options are not samples",
        },
        "claim_boundary": "This compares interfaces for realised candidate ranking in one evidence-grounded finite-option condition. It does not establish universal detector ranking, pre-generation branch prediction, or causal localisation.",
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {output}")
    for name, record in payload["internal_interfaces"].items():
        print(
            f"{name}: AUROC={record['within_question_auroc']['mean']:.3f}; "
            f"selection={record['selection_accuracy']['mean']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
