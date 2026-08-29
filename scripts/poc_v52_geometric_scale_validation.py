"""V52 journal-scale geometric validation inspired by CLUE activation deltas.

The script evaluates fixed source-to-target centroid geometry on three models
and two tasks. It keeps question groups isolated, reports exact within-question
discrimination, compares delta/final-state/norm/confidence controls, and audits
sample-size stability. Target labels are never used to fit deployed scores.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_nn32_train_qwen7b_true_sae import TopKSAE


SEED = 20260828
N_TARGET_BOOTSTRAP = 1000
N_SUBSAMPLE_REPEATS = 30
STAGES = (0, 1, 2, 3)
PRIMARY_STAGE = 3
RESULT = ROOT / "outputs" / "poc_v52_geometric_scale_validation.json"
TARGET_SAE_CACHE = ROOT / "outputs" / "cache" / "nn34_qwen7b_true_sae_states.npz"
QWEN7B_SAE = ROOT / "outputs" / "nn32_qwen7b_l18_topk_sae.pt"

CONFIGS = {
    "Q7-WQ": {
        "model": "Qwen2.5-7B-Instruct NF4",
        "task": "WebQuestions",
        "source": ROOT / "outputs/cache/nn32_qwen7b_webquestions_true_sae_states.npz",
        "target": ROOT / "outputs/cache/nn34_independent_webquestions_trajectory_states.npz",
        "external": True,
        "representations": ("raw", "sae"),
    },
    "G2-TQA": {
        "model": "Gemma-2-2B",
        "task": "TriviaQA",
        "source": ROOT / "outputs/cache/fpe14_gemma_trivia_confirmatory_trajectory_states.npz",
        "target": None,
        "external": False,
        "representations": ("raw", "sae"),
    },
    "Q1-TQA": {
        "model": "Qwen2.5-1.5B",
        "task": "TriviaQA",
        "source": ROOT / "outputs/cache/fpe14_qwen15_trivia_confirmatory_trajectory_states.npz",
        "target": None,
        "external": False,
        "representations": ("raw",),
    },
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return {key: archive[key] for key in archive.files}


def ensure_qwen7b_target_sae(target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if TARGET_SAE_CACHE.exists():
        archive = np.load(TARGET_SAE_CACHE, allow_pickle=True)
        for stage in STAGES:
            target[f"sae_T{stage}"] = archive[f"sae_T{stage}"]
        return target
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required once to encode the Qwen-7B target SAE cache")
    checkpoint = torch.load(QWEN7B_SAE, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    sae = TopKSAE(
        config["d_model"], config["d_sae"], config["top_k"], checkpoint["state_dict"]["mean"]
    ).cuda().eval()
    sae.load_state_dict(checkpoint["state_dict"])
    encoded = {}
    with torch.inference_mode():
        for stage in STAGES:
            values = target[f"raw_T{stage}"].astype(np.float32)
            blocks = []
            for start in range(0, len(values), 256):
                latent = sae.encode(torch.as_tensor(values[start : start + 256], device="cuda"))
                blocks.append(latent.cpu().numpy().astype(np.float16))
            encoded[f"sae_T{stage}"] = np.concatenate(blocks)
            target[f"sae_T{stage}"] = encoded[f"sae_T{stage}"]
    np.savez_compressed(TARGET_SAE_CACHE, **encoded)
    return target


def stable_half_split(states: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    question_ids = states["question_ids"].astype(str)
    questions = np.unique(question_ids)
    order = sorted(
        questions,
        key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest(),
    )
    source_questions = set(order[: len(order) // 2])
    source_mask = np.asarray([value in source_questions for value in question_ids])
    target_mask = ~source_mask
    row_items = {
        key: value
        for key, value in states.items()
        if getattr(value, "ndim", 0) > 0 and value.shape[0] == len(question_ids)
    }
    return (
        {key: value[source_mask] for key, value in row_items.items()},
        {key: value[target_mask] for key, value in row_items.items()},
    )


def valid_rows(states: dict[str, np.ndarray], stage: int) -> np.ndarray:
    key = f"valid_T{stage}"
    return states[key].astype(bool) if key in states else np.ones(len(states["labels"]), dtype=bool)


def features(states: dict[str, np.ndarray], representation: str, stage: int, mode: str) -> np.ndarray:
    current = states[f"{representation}_T{stage}"].astype(np.float32)
    if mode == "absolute":
        return current
    if mode == "delta":
        return current - states[f"{representation}_T0"].astype(np.float32)
    raise ValueError(mode)


def centroid_model(x: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    correct = x[labels == 0].mean(axis=0, dtype=np.float64).astype(np.float32)
    error = x[labels == 1].mean(axis=0, dtype=np.float64).astype(np.float32)
    return {"correct": correct, "error": error, "direction": error - correct}


def centroid_score(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    # ||x-mu_correct||^2 - ||x-mu_error||^2, expanded to avoid large matrices.
    direction = model["direction"].astype(np.float64)
    intercept = float(np.dot(model["correct"], model["correct"]) - np.dot(model["error"], model["error"]))
    return (2.0 * x.astype(np.float64) @ direction + intercept).astype(np.float64)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def within_question_auc(labels: np.ndarray, scores: np.ndarray, question_ids: np.ndarray) -> float:
    values = []
    for question in np.unique(question_ids):
        local = question_ids == question
        correct = scores[local & (labels == 0)]
        error = scores[local & (labels == 1)]
        if len(correct) == 0 or len(error) == 0:
            continue
        comparisons = error[:, None] - correct[None, :]
        values.append(float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0))))
    return float(np.mean(values)) if values else float("nan")


def center_by_question(x: np.ndarray, question_ids: np.ndarray) -> np.ndarray:
    output = np.empty_like(x, dtype=np.float32)
    for question in np.unique(question_ids):
        local = question_ids == question
        output[local] = x[local] - x[local].mean(axis=0, keepdims=True)
    return output


def metric_row(labels: np.ndarray, scores: np.ndarray, question_ids: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "within_question_auroc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "within_question_auroc": within_question_auc(labels, scores, question_ids),
    }


def grouped_bootstrap(
    labels: np.ndarray, scores: np.ndarray, question_ids: np.ndarray, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    questions = np.unique(question_ids)
    groups = [np.flatnonzero(question_ids == question) for question in questions]
    group_within = []
    for rows in groups:
        correct = scores[rows][labels[rows] == 0]
        error = scores[rows][labels[rows] == 1]
        if len(correct) == 0 or len(error) == 0:
            group_within.append(float("nan"))
        else:
            comparisons = error[:, None] - correct[None, :]
            group_within.append(float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0))))
    group_within = np.asarray(group_within, dtype=np.float64)
    estimates = []
    for _ in range(N_TARGET_BOOTSTRAP):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y = np.concatenate([labels[groups[index]] for index in sampled])
        s = np.concatenate([scores[groups[index]] for index in sampled])
        if len(np.unique(y)) == 2:
            estimates.append((roc_auc_score(y, s), float(np.nanmean(group_within[sampled]))))
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "auroc_95ci": [float(np.quantile(values[:, 0], 0.025)), float(np.quantile(values[:, 0], 0.975))],
        "within_question_auroc_95ci": [
            float(np.nanquantile(values[:, 1], 0.025)),
            float(np.nanquantile(values[:, 1], 0.975)),
        ],
        "n_bootstrap": int(len(values)),
    }


def fit_confidence(source: dict[str, np.ndarray], stage: int, train: np.ndarray) -> dict:
    x = np.concatenate(
        (source["confidence_T0"].astype(np.float32), source[f"confidence_T{stage}"].astype(np.float32)),
        axis=1,
    )
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale < 1e-6] = 1.0
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    model.fit((x[train] - mean) / scale, source["labels"][train].astype(np.int8))
    return {"model": model, "mean": mean, "scale": scale}


def predict_confidence(model: dict, target: dict[str, np.ndarray], stage: int) -> np.ndarray:
    x = np.concatenate(
        (target["confidence_T0"].astype(np.float32), target[f"confidence_T{stage}"].astype(np.float32)),
        axis=1,
    )
    return model["model"].predict_proba((x - model["mean"]) / model["scale"])[:, 1]


def random_label_control(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_q: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_q: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(30):
        shuffled = source_y.copy()
        for question in np.unique(source_q):
            local = np.flatnonzero(source_q == question)
            shuffled[local] = rng.permutation(shuffled[local])
        if len(np.unique(shuffled)) < 2:
            continue
        score = centroid_score(centroid_model(source_x, shuffled), target_x)
        rows.append(metric_row(target_y, score, target_q))
    return {
        "auroc_mean": float(np.mean([row["auroc"] for row in rows])),
        "within_question_auroc_mean": float(np.nanmean([row["within_question_auroc"] for row in rows])),
        "n_permutations": len(rows),
    }


def sample_efficiency(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    representation: str,
    stage: int,
    seed: int,
) -> list[dict]:
    source_valid = valid_rows(source, stage)
    target_valid = valid_rows(target, stage)
    source_q = source["question_ids"].astype(str)
    target_q = target["question_ids"].astype(str)[target_valid]
    source_y = source["labels"].astype(np.int8)
    target_y = target["labels"].astype(np.int8)[target_valid]
    source_x = features(source, representation, stage, "delta")
    target_x = features(target, representation, stage, "delta")[target_valid]
    source_x_centered = source_x.copy()
    source_x_centered[source_valid] = center_by_question(source_x[source_valid], source_q[source_valid])
    target_x = center_by_question(target_x, target_q)
    full = centroid_model(source_x_centered[source_valid], source_y[source_valid])
    questions = np.unique(source_q[source_valid])
    sizes = sorted({value for value in (16, 32, 64, 128, 256, 480) if value <= len(questions)})
    output = []
    for size in sizes:
        aucs, within, direction_cosines = [], [], []
        for repeat in range(N_SUBSAMPLE_REPEATS):
            rng = np.random.default_rng(seed + size * 100 + repeat)
            selected = rng.choice(questions, size=size, replace=False)
            train = source_valid & np.isin(source_q, selected)
            if len(np.unique(source_y[train])) < 2:
                continue
            local = centroid_model(source_x_centered[train], source_y[train])
            score = centroid_score(local, target_x)
            row = metric_row(target_y, score, target_q)
            aucs.append(row["auroc"])
            within.append(row["within_question_auroc"])
            direction_cosines.append(cosine(local["direction"], full["direction"]))
        output.append(
            {
                "source_questions": size,
                "approx_source_trajectories": int(size * len(source_y) / len(np.unique(source_q))),
                "n_repeats": len(aucs),
                "target_auroc_median": float(np.median(aucs)),
                "target_auroc_10_90": [float(np.quantile(aucs, 0.1)), float(np.quantile(aucs, 0.9))],
                "within_question_auroc_median": float(np.nanmedian(within)),
                "direction_cosine_to_full_median": float(np.nanmedian(direction_cosines)),
                "direction_cosine_10_90": [
                    float(np.nanquantile(direction_cosines, 0.1)),
                    float(np.nanquantile(direction_cosines, 0.9)),
                ],
            }
        )
    return output


def evaluate_condition(name: str, config: dict, source: dict, target: dict) -> dict:
    source_y = source["labels"].astype(np.int8)
    target_y_all = target["labels"].astype(np.int8)
    source_q = source["question_ids"].astype(str)
    target_q_all = target["question_ids"].astype(str)
    output = {
        "model": config["model"],
        "task": config["task"],
        "confirmation_type": "question-disjoint external" if config["external"] else "fixed grouped internal",
        "source_questions": int(len(np.unique(source_q))),
        "source_trajectories": int(len(source_y)),
        "target_questions": int(len(np.unique(target_q_all))),
        "target_trajectories": int(len(target_y_all)),
        "source_error_rate": float(source_y.mean()),
        "target_error_rate": float(target_y_all.mean()),
        "representations": {},
    }
    for representation in config["representations"]:
        stage_results = {}
        for stage in STAGES:
            source_valid = valid_rows(source, stage)
            target_valid = valid_rows(target, stage)
            source_labels = source_y[source_valid]
            target_labels = target_y_all[target_valid]
            target_q = target_q_all[target_valid]
            rows = {}
            for mode in ("delta", "absolute"):
                source_x = features(source, representation, stage, mode)[source_valid]
                target_x = features(target, representation, stage, mode)[target_valid]
                model = centroid_model(source_x, source_labels)
                score = centroid_score(model, target_x)
                local = metric_row(target_labels, score, target_q)
                local.update(grouped_bootstrap(target_labels, score, target_q, SEED + stage * 101 + len(name)))
                target_direction = centroid_model(target_x, target_labels)["direction"]
                local["source_target_direction_cosine"] = cosine(model["direction"], target_direction)
                rows[mode] = local
            source_delta = features(source, representation, stage, "delta")[source_valid]
            target_delta = features(target, representation, stage, "delta")[target_valid]
            source_centered = center_by_question(source_delta, source_q[source_valid])
            target_centered = center_by_question(target_delta, target_q)
            centered_model = centroid_model(source_centered, source_labels)
            centered_score = centroid_score(centered_model, target_centered)
            centered_row = metric_row(target_labels, centered_score, target_q)
            centered_row.update(
                grouped_bootstrap(target_labels, centered_score, target_q, SEED + stage * 131 + len(name))
            )
            centered_target_direction = centroid_model(target_centered, target_labels)["direction"]
            centered_row["source_target_direction_cosine"] = cosine(
                centered_model["direction"], centered_target_direction
            )
            rows["question_centered_delta"] = centered_row
            norm_score = np.linalg.norm(target_delta, axis=1)
            rows["delta_norm"] = metric_row(target_labels, norm_score, target_q)
            if representation == "raw":
                confidence_model = fit_confidence(source, stage, source_valid)
                confidence_score = predict_confidence(confidence_model, target, stage)[target_valid]
                rows["confidence"] = metric_row(target_labels, confidence_score, target_q)
            if stage == PRIMARY_STAGE:
                rows["random_label_control"] = random_label_control(
                    source_centered,
                    source_labels,
                    source_q[source_valid],
                    target_centered,
                    target_labels,
                    target_q,
                    SEED + len(name),
                )
            stage_results[f"T{stage}"] = rows
        output["representations"][representation] = {
            "stages": stage_results,
            "sample_efficiency_T3": sample_efficiency(
                source, target, representation, PRIMARY_STAGE, SEED + len(name) + len(representation)
            ),
        }
    return output


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    payload = {
        "experiment": "V52 journal-scale CLUE-inspired geometric readout validation",
        "protocol": "paper/V52_GEOMETRIC_SCALE_PROTOCOL.md",
        "reference_geometry": (
            "Activation delta, success/failure centroids, direction transfer, and experience-size stability "
            "adapted from Liang et al., ACL 2026. This is a staged single-layer analogue, not an all-layer reproduction."
        ),
        "results": {},
    }
    for name, config in CONFIGS.items():
        print(f"loading {name}", flush=True)
        loaded = load_npz(config["source"])
        if config["target"] is None:
            source, target = stable_half_split(loaded)
        else:
            source, target = loaded, load_npz(config["target"])
        if name == "Q7-WQ":
            target = ensure_qwen7b_target_sae(target)
        print(
            f"{name}: source={len(source['labels'])}, target={len(target['labels'])}, "
            f"representations={config['representations']}",
            flush=True,
        )
        payload["results"][name] = evaluate_condition(name, config, source, target)
    payload["total_unique_questions"] = int(
        sum(row["source_questions"] + row["target_questions"] for row in payload["results"].values())
    )
    payload["total_trajectories"] = int(
        sum(row["source_trajectories"] + row["target_trajectories"] for row in payload["results"].values())
    )
    payload["limitations"] = (
        "Only one cached layer is available, whereas CLUE aggregates all layers across a full reasoning span. "
        "The small-model TriviaQA evaluations are grouped internal replications. Qwen-7B WebQuestions is the "
        "only question-disjoint external confirmation."
    )
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {RESULT}", flush=True)
    for name, row in payload["results"].items():
        for representation, values in row["representations"].items():
            t3 = values["stages"]["T3"]
            print(
                f"{name}/{representation}: delta AUROC={t3['delta']['auroc']:.3f}, "
                f"within={t3['delta']['within_question_auroc']:.3f}, "
                f"direction cosine={t3['delta']['source_target_direction_cosine']:.3f}, "
                f"centered AUROC={t3['question_centered_delta']['auroc']:.3f}, "
                f"centered cosine={t3['question_centered_delta']['source_target_direction_cosine']:.3f}, "
                f"absolute AUROC={t3['absolute']['auroc']:.3f}, norm={t3['delta_norm']['auroc']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
