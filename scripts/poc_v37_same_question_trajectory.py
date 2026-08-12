"""V37: same-question stochastic trajectory divergence pilot.

For each fixed evidence question, sample multiple completions, extract T0-T5
SAE/residual states, and evaluate population and question-centered probes with
outer folds grouped by question ID. T0 is a required negative control for
realized-outcome prediction within a discordant question.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.poc_v35_factorial_pilot import build_pairs
from src.extract import ResidualHook
from src.load import load_all


LAYERS = [9, 18]
STAGES = [f"T{k}" for k in range(6)]
SEEDS = [13, 29, 42]
TOKEN_HASH_DIM = 1024


def paths(tag: str) -> tuple[Path, Path, Path]:
    return (
        ROOT / "data" / f"{tag}_2wiki_trajectory.jsonl",
        ROOT / "outputs" / "cache" / f"{tag}_2wiki_trajectory_states.npz",
        ROOT / "outputs" / f"poc_{tag}_same_question_trajectory_results.json",
    )


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(response: str, gold: str) -> float:
    response_tokens = normalize_answer(response).split()
    gold_tokens = normalize_answer(gold).split()
    if not response_tokens or not gold_tokens:
        return 0.0
    common = Counter(response_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(response_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(response: str, gold: str) -> tuple[bool, bool, float]:
    exact = normalize_answer(response) == normalize_answer(gold)
    f1 = token_f1(response, gold)
    return bool(exact or f1 >= 0.90), bool(exact), float(f1)


def build_prompt(pair: dict, difficulty: str) -> str:
    triples = pair["evidences"][-1:] if difficulty == "simple" else pair["evidences"]
    facts = "\n".join(
        f"{idx}. {str(subject).strip()} | {str(relation).strip()} | {str(obj).strip()}"
        for idx, (subject, relation, obj) in enumerate(triples, 1)
    )
    question = pair[f"{difficulty}_question"]
    return (
        "Answer using only the facts below. Return only the shortest exact answer "
        "span copied verbatim from the facts. Do not explain or rephrase.\n"
        f"Facts:\n{facts}\nQuestion: {question}\nAnswer:"
    )


def clean_generated_ids(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    clean = []
    for token_id in token_ids:
        if token_id in stop_ids:
            break
        clean.append(int(token_id))
    return clean


def generate_rows(
    assets,
    pairs: list[dict],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    rows = []
    stop_ids = {int(assets.tokenizer.eos_token_id)}
    eos_ids = sorted(stop_ids)

    prompts = [
        (pair, difficulty, build_prompt(pair, difficulty))
        for pair in pairs
        for difficulty in ["simple", "complex"]
    ]
    for prompt_idx, (pair, difficulty, prompt) in enumerate(tqdm(prompts, desc="v37-sample-prompts")):
        inputs = assets.tokenizer(prompt, return_tensors="pt").to(assets.device)
        torch.manual_seed(seed + prompt_idx)
        with torch.no_grad():
            generated = assets.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n_samples,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=assets.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        question_id = f"{pair['pair_id']}::{difficulty}"
        for sample_idx, sequence in enumerate(generated):
            token_ids = clean_generated_ids(
                sequence[prompt_len:].detach().cpu().tolist(), stop_ids
            )
            decoded = assets.tokenizer.decode(token_ids, skip_special_tokens=True)
            nonempty_lines = [line.strip() for line in decoded.splitlines() if line.strip()]
            response = nonempty_lines[0] if nonempty_lines else decoded.strip()
            response = re.sub(r"^answer\s*:\s*", "", response, flags=re.IGNORECASE).strip()
            correct, exact, f1 = score_answer(response, pair["gold_answer"])
            rows.append({
                "pair_id": pair["pair_id"],
                "question_id": question_id,
                "difficulty": difficulty,
                "sample_index": sample_idx,
                "prompt": prompt,
                "gold_answer": pair["gold_answer"],
                "model_answer": response,
                "generated_token_ids": token_ids,
                "model_correct": correct,
                "normalized_exact_match": exact,
                "token_f1": f1,
            })
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution_features(logits: torch.Tensor) -> tuple[float, float, float]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    top2 = torch.topk(probs, k=2).values
    entropy = -(probs * log_probs).sum()
    return (
        float(top2[0].detach().cpu()),
        float(entropy.detach().cpu()),
        float((top2[0] - top2[1]).detach().cpu()),
    )


def prefix_hash(token_ids: list[int], stage: int) -> np.ndarray:
    features = np.zeros(TOKEN_HASH_DIM, dtype=np.float32)
    for position, token_id in enumerate(token_ids[:stage]):
        index = (int(token_id) * 1009 + position * 9176) % TOKEN_HASH_DIM
        features[index] += 1.0
    return features


def extract_states(assets, rows: list[dict], cache_path: Path) -> dict:
    hooks = {
        layer: ResidualHook().attach(assets.model.model.layers[layer])
        for layer in LAYERS
    }
    n_rows = len(rows)
    sae = {stage: {layer: [None] * n_rows for layer in LAYERS} for stage in STAGES}
    raw = {stage: {layer: [None] * n_rows for layer in LAYERS} for stage in STAGES}
    confidence = {stage: [None] * n_rows for stage in STAGES}
    token_prefix = {stage: [None] * n_rows for stage in STAGES}
    valid = {stage: np.zeros(n_rows, dtype=bool) for stage in STAGES}
    generated_by_row = [
        [int(token_id) for token_id in row["generated_token_ids"]]
        for row in rows
    ]
    selected_log_probs = [[] for _ in rows]
    question_rows = {}
    for row_idx, row in enumerate(rows):
        question_rows.setdefault(row["question_id"], []).append(row_idx)
        for stage_idx, stage in enumerate(STAGES):
            valid[stage][row_idx] = stage_idx == 0 or len(generated_by_row[row_idx]) >= stage_idx
            token_prefix[stage][row_idx] = prefix_hash(generated_by_row[row_idx], stage_idx)

    try:
        iterator = tqdm(question_rows.items(), desc="v37-extract-prefix-trajectories")
        for _, row_indices in iterator:
            prompt_inputs = assets.tokenizer(
                rows[row_indices[0]]["prompt"], return_tensors="pt"
            ).to(assets.device)
            prompt_ids = prompt_inputs["input_ids"]

            for stage_idx, stage in enumerate(STAGES):
                active = [row_idx for row_idx in row_indices if valid[stage][row_idx]]
                if not active:
                    continue
                if stage_idx == 0:
                    stage_ids = prompt_ids
                else:
                    prefixes = [
                        torch.tensor(
                            generated_by_row[row_idx][:stage_idx],
                            dtype=prompt_ids.dtype,
                            device=assets.device,
                        )
                        for row_idx in active
                    ]
                    prefix_batch = torch.stack(prefixes)
                    stage_ids = torch.cat([
                        prompt_ids.expand(len(active), -1), prefix_batch
                    ], dim=1)

                with torch.no_grad():
                    outputs = assets.model(
                        input_ids=stage_ids,
                        attention_mask=torch.ones_like(stage_ids),
                        use_cache=False,
                    )
                stage_logits = outputs.logits[:, -1]

                if stage_idx == 0:
                    max_prob, entropy, margin = distribution_features(stage_logits[0])
                    base_confidence = np.asarray(
                        [max_prob, entropy, margin, 0.0, 0.0], dtype=np.float32
                    )
                    for row_idx in active:
                        confidence[stage][row_idx] = base_confidence.copy()
                    for layer in LAYERS:
                        vector = hooks[layer].value[0, -1]
                        raw_value = vector.float().cpu().numpy().astype(np.float16)
                        with torch.no_grad():
                            latent = assets.saes[layer].encode(vector.unsqueeze(0)).squeeze(0)
                        sae_value = latent.float().cpu().numpy().astype(np.float16)
                        for row_idx in active:
                            raw[stage][layer][row_idx] = raw_value.copy()
                            sae[stage][layer][row_idx] = sae_value.copy()
                else:
                    for batch_idx, row_idx in enumerate(active):
                        max_prob, entropy, margin = distribution_features(stage_logits[batch_idx])
                        history = selected_log_probs[row_idx]
                        confidence[stage][row_idx] = np.asarray([
                            max_prob,
                            entropy,
                            margin,
                            float(np.mean(history)) if history else 0.0,
                            float(np.min(history)) if history else 0.0,
                        ], dtype=np.float32)
                    for layer in LAYERS:
                        vectors = hooks[layer].value[:, -1]
                        with torch.no_grad():
                            latents = assets.saes[layer].encode(vectors)
                        raw_values = vectors.float().cpu().numpy().astype(np.float16)
                        sae_values = latents.float().cpu().numpy().astype(np.float16)
                        for batch_idx, row_idx in enumerate(active):
                            raw[stage][layer][row_idx] = raw_values[batch_idx]
                            sae[stage][layer][row_idx] = sae_values[batch_idx]

                for batch_idx, row_idx in enumerate(active):
                    if len(generated_by_row[row_idx]) > stage_idx:
                        next_token = generated_by_row[row_idx][stage_idx]
                        logits_idx = 0 if stage_idx == 0 else batch_idx
                        log_prob = torch.log_softmax(stage_logits[logits_idx].float(), -1)[next_token]
                        selected_log_probs[row_idx].append(float(log_prob.detach().cpu()))
    finally:
        for hook in hooks.values():
            hook.detach()

    hidden_size = int(assets.model.config.hidden_size)
    for stage in STAGES:
        for row_idx in range(n_rows):
            if confidence[stage][row_idx] is None:
                confidence[stage][row_idx] = np.zeros(5, dtype=np.float32)
            for layer in LAYERS:
                if raw[stage][layer][row_idx] is None:
                    raw[stage][layer][row_idx] = np.zeros(hidden_size, dtype=np.float16)
                if sae[stage][layer][row_idx] is None:
                    sae[stage][layer][row_idx] = np.zeros(
                        assets.saes[layer].cfg.d_sae, dtype=np.float16
                    )

    payload = {
        "labels": np.asarray([0 if row["model_correct"] else 1 for row in rows], dtype=np.int8),
        "question_ids": np.asarray([row["question_id"] for row in rows], dtype=object),
        "pair_ids": np.asarray([row["pair_id"] for row in rows], dtype=object),
        "difficulty": np.asarray([row["difficulty"] for row in rows], dtype=object),
        "answer_lengths": np.asarray([len(row["generated_token_ids"]) for row in rows], dtype=np.int16),
    }
    for stage in STAGES:
        payload[f"valid_{stage}"] = valid[stage]
        payload[f"confidence_{stage}"] = np.stack(confidence[stage])
        payload[f"token_prefix_{stage}"] = np.stack(token_prefix[stage])
        for layer in LAYERS:
            payload[f"raw_{stage}_L{layer}"] = np.stack(raw[stage][layer])
            payload[f"sae_{stage}_L{layer}"] = np.stack(sae[stage][layer])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def load_states(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    return {key: arr[key] for key in arr.files}


def center_within_question(X: np.ndarray, question_ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    centered = np.zeros_like(X, dtype=np.float32)
    for question_id in np.unique(question_ids[valid]):
        mask = valid & (question_ids == question_id)
        block = X[mask].astype(np.float32)
        if np.all(block == block[0]):
            centered[mask] = 0.0
        else:
            centered[mask] = block - block.mean(axis=0, keepdims=True)
    return centered


def make_classifier(kind: str, seed: int) -> LogisticRegression:
    if kind == "sae":
        return LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1, class_weight="balanced",
            max_iter=3000, random_state=seed,
        )
    return LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0, class_weight="balanced",
        max_iter=3000, random_state=seed,
    )


def fit_predict(X_train, y_train, X_test, kind: str, seed: int) -> np.ndarray:
    sparse_style = kind == "sae"
    scaler = StandardScaler(with_mean=not sparse_style)
    Xtr = scaler.fit_transform(X_train.astype(np.float64))
    Xte = scaler.transform(X_test.astype(np.float64))
    clf = make_classifier(kind, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    question_ids: np.ndarray,
    valid: np.ndarray,
    kind: str,
    centered: bool,
) -> list[dict]:
    if int(valid.sum()) == 0 or len(np.unique(y[valid])) < 2:
        return []
    X_eval = center_within_question(X, question_ids, valid) if centered else X.astype(np.float32)
    unique_questions = np.unique(question_ids[valid])
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = unique_questions.copy()
        rng.shuffle(shuffled)
        fold_map = {question_id: idx % 5 for idx, question_id in enumerate(shuffled)}
        predictions = np.full(len(y), np.nan)
        for fold in range(5):
            test = valid & np.asarray([fold_map.get(question_id, -1) == fold for question_id in question_ids])
            train = valid & ~np.asarray([fold_map.get(question_id, -1) == fold for question_id in question_ids])
            if len(np.unique(y[train])) < 2 or not np.any(test):
                continue
            predictions[test] = fit_predict(X_eval[train], y[train], X_eval[test], kind, seed + fold)

        scored = valid & np.isfinite(predictions)
        if int(scored.sum()) == 0 or len(np.unique(y[scored])) < 2:
            continue
        population_auroc = float(roc_auc_score(y[scored], predictions[scored]))
        population_auprc = float(average_precision_score(y[scored], predictions[scored]))
        question_aurocs = []
        for question_id in np.unique(question_ids[scored]):
            mask = scored & (question_ids == question_id)
            if len(np.unique(y[mask])) == 2:
                local_scores = predictions[mask]
                if float(np.ptp(local_scores)) <= 1e-10:
                    question_aurocs.append(0.5)
                else:
                    question_aurocs.append(float(roc_auc_score(y[mask], local_scores)))
        rows.append({
            "seed": seed,
            "representation": "question_centered" if centered else "population",
            "population_auroc": population_auroc,
            "population_auprc": population_auprc,
            "within_question_auroc_macro": float(np.mean(question_aurocs)) if question_aurocs else None,
            "within_question_auroc_std": float(np.std(question_aurocs, ddof=1)) if len(question_aurocs) > 1 else None,
            "n": int(scored.sum()),
            "n_discordant_questions": int(len(question_aurocs)),
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({
        (row["stage"], row["layer"], row["method"], row["representation"])
        for row in rows
    }, key=str)
    output = []
    for stage, layer, method, representation in keys:
        selected = [
            row for row in rows
            if (row["stage"], row["layer"], row["method"], row["representation"])
            == (stage, layer, method, representation)
        ]
        output.append({
            "stage": stage,
            "layer": layer,
            "method": method,
            "representation": representation,
            "n_seeds": len(selected),
            "population_auroc_mean": float(np.mean([row["population_auroc"] for row in selected])),
            "population_auroc_std": float(np.std([row["population_auroc"] for row in selected], ddof=1)),
            "population_auprc_mean": float(np.mean([row["population_auprc"] for row in selected])),
            "within_question_auroc_macro_mean": (
                float(np.mean([row["within_question_auroc_macro"] for row in selected]))
                if selected[0]["within_question_auroc_macro"] is not None else None
            ),
            "within_question_auroc_macro_std": (
                float(np.std([row["within_question_auroc_macro"] for row in selected], ddof=1))
                if selected[0]["within_question_auroc_macro"] is not None else None
            ),
            "n": selected[0]["n"],
            "n_discordant_questions": selected[0]["n_discordant_questions"],
        })
    return output


def question_statistics(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["question_id"], []).append(row)
    discordant = {
        question_id for question_id, samples in grouped.items()
        if len({sample["model_correct"] for sample in samples}) == 2
    }
    by_difficulty = {}
    for difficulty in ["simple", "complex"]:
        selected = [row for row in rows if row["difficulty"] == difficulty]
        selected_questions = {row["question_id"] for row in selected}
        by_difficulty[difficulty] = {
            "n_questions": len(selected_questions),
            "n_completions": len(selected),
            "accuracy": float(np.mean([row["model_correct"] for row in selected])),
            "discordant_questions": len(selected_questions & discordant),
        }
    return {
        "n_questions": len(grouped),
        "n_discordant_questions": len(discordant),
        "discordant_fraction": len(discordant) / max(len(grouped), 1),
        "by_difficulty": by_difficulty,
        "answer_length_mean": float(np.mean([len(row["generated_token_ids"]) for row in rows])),
        "answer_length_median": float(np.median([len(row["generated_token_ids"]) for row in rows])),
    }


def t0_variance(states: dict) -> dict:
    question_ids = states["question_ids"].astype(str)
    output = {}
    for layer in LAYERS:
        for method in ["raw", "sae"]:
            X = states[f"{method}_T0_L{layer}"].astype(np.float32)
            maxima = []
            for question_id in np.unique(question_ids):
                block = X[question_ids == question_id]
                maxima.append(float(np.max(np.var(block, axis=0))))
            output[f"{method}_L{layer}"] = {
                "max_within_question_feature_variance": float(np.max(maxima)),
                "mean_of_question_max_variance": float(np.mean(maxima)),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="v37")
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--samples-per-question", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    data_path, cache_path, out_path = paths(args.tag)
    assets = None
    if args.force_generate or not data_path.exists():
        pairs = build_pairs(args.n_pairs, args.seed)
        assets = load_all(args.config)
        rows = generate_rows(
            assets, pairs, args.samples_per_question, args.temperature,
            args.top_p, args.max_new_tokens, args.seed,
        )
        save_rows(rows, data_path)
        print(f"[v37] saved generations -> {data_path}")
    else:
        rows = load_rows(data_path)
        print(f"[v37] loaded generations -> {data_path}")

    stats = question_statistics(rows)
    print(f"[v37] question stats: {stats}")
    if args.force_extract or not cache_path.exists():
        if assets is None:
            assets = load_all(args.config)
        states = extract_states(assets, rows, cache_path)
        print(f"[v37] saved states -> {cache_path}")
    else:
        states = load_states(cache_path)
        print(f"[v37] loaded states -> {cache_path}")

    y = states["labels"].astype(int)
    question_ids = states["question_ids"].astype(str)
    eval_rows = []
    for stage in STAGES:
        valid = states[f"valid_{stage}"].astype(bool)
        feature_sets = [
            ("confidence", None, states[f"confidence_{stage}"], "raw"),
            ("token_prefix", None, states[f"token_prefix_{stage}"], "raw"),
        ]
        for layer in LAYERS:
            raw_features = states[f"raw_{stage}_L{layer}"]
            sae_features = states[f"sae_{stage}_L{layer}"]
            token_features = states[f"token_prefix_{stage}"]
            feature_sets.extend([
                ("raw_residual", layer, raw_features, "raw"),
                ("sae", layer, sae_features, "sae"),
                (
                    "raw_token_fused", layer,
                    np.hstack([raw_features, token_features]), "raw",
                ),
                (
                    "sae_token_fused", layer,
                    np.hstack([sae_features, token_features]), "sae",
                ),
            ])
        for method, layer, X, kind in feature_sets:
            for centered in [False, True]:
                result_rows = evaluate(
                    X.astype(np.float32), y, question_ids, valid, kind, centered,
                )
                eval_rows.extend({
                    "stage": stage,
                    "layer": layer,
                    "method": method,
                    **row,
                } for row in result_rows)

    summaries = summarize(eval_rows)
    summaries.sort(
        key=lambda row: (
            row["within_question_auroc_macro_mean"]
            if row["within_question_auroc_macro_mean"] is not None else -1
        ),
        reverse=True,
    )
    payload = {
        "experiment": "poc_v37_same_question_trajectory",
        "model": "google/gemma-2-2b",
        "dataset": "2Wiki matched evidence questions",
        "n_pairs": args.n_pairs,
        "samples_per_question": args.samples_per_question,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "label": "normalized exact match OR token F1 >= 0.90",
        "question_statistics": stats,
        "stage_valid_counts": {
            stage: int(states[f"valid_{stage}"].sum()) for stage in STAGES
        },
        "t0_negative_control": t0_variance(states),
        "summary": summaries,
        "rows": eval_rows,
        "data_path": str(data_path),
        "cache_path": str(cache_path),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v37] saved results -> {out_path}")
    for row in summaries[:25]:
        print(
            f"[v37] {row['stage']} {row['method']} L{row['layer']} "
            f"{row['representation']} pop={row['population_auroc_mean']:.4f} "
            f"within={row['within_question_auroc_macro_mean']} "
            f"n={row['n']} q={row['n_discordant_questions']}"
        )


if __name__ == "__main__":
    main()
