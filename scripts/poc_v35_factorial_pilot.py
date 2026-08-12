"""PoC V35: matched simple/complex factorial hallucination-state pilot.

Build matched pairs from 2Wiki compositional/inference evidence chains:
  - complex: the original multi-hop question
  - simple: a direct question about the final evidence triple

Both conditions share the same gold answer. The script generates greedy model
answers, extracts SAE/raw residual states at T0 (final prompt token) and T1
(first generated token), and evaluates grouped cross-difficulty transfer.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.self_knowledge import FEW_SHOT_TEMPLATE, is_answer_correct
from src.extract import ResidualHook
from src.load import load_all


OUT_PATH = ROOT / "outputs" / "poc_v35_factorial_pilot_results.json"
DATA_PATH = ROOT / "data" / "v35_2wiki_factorial_pilot.jsonl"
CACHE_PATH = ROOT / "outputs" / "cache" / "v35_2wiki_factorial_states.npz"
CONTEXT_OUT_PATH = ROOT / "outputs" / "poc_v35b_factorial_context_pilot_results.json"
CONTEXT_DATA_PATH = ROOT / "data" / "v35b_2wiki_factorial_context_pilot.jsonl"
CONTEXT_CACHE_PATH = ROOT / "outputs" / "cache" / "v35b_2wiki_factorial_context_states.npz"
LAYERS = [9, 12, 18, 20]
SEEDS = [13, 29, 42, 71, 101]

RELATION_TEMPLATES = {
    "director": "Who directed {subject}?",
    "date of birth": "When was {subject} born?",
    "father": "Who is the father of {subject}?",
    "date of death": "When did {subject} die?",
    "publication date": "When was {subject} released?",
    "country of citizenship": "What country is {subject} a citizen of?",
    "place of birth": "Where was {subject} born?",
    "spouse": "Who is the spouse of {subject}?",
    "mother": "Who is the mother of {subject}?",
    "place of death": "Where did {subject} die?",
    "country of origin": "What is the country of origin of {subject}?",
    "country": "In which country is {subject} located?",
    "performer": "Who performed {subject}?",
    "composer": "Who composed {subject}?",
    "educated at": "Where was {subject} educated?",
    "place of burial": "Where was {subject} buried?",
    "employer": "Who employed {subject}?",
    "inception": "When was {subject} established?",
    "award received": "What award did {subject} receive?",
    "child": "Who is a child of {subject}?",
    "sibling": "Who is a sibling of {subject}?",
    "cause of death": "What was the cause of death of {subject}?",
    "founded by": "Who founded {subject}?",
    "producer": "Who produced {subject}?",
    "publisher": "Who published {subject}?",
    "occupation": "What is the occupation of {subject}?",
    "creator": "Who created {subject}?",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return bool(na and nb and (na == nb or na in nb or nb in na))


def build_pairs(n_pairs: int, seed: int) -> List[dict]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(
        "voidful/2WikiMultihopQA",
        split="validation",
        cache_dir=str(ROOT / "data" / "cache"),
    )
    candidates = []
    for row in ds:
        if str(row.get("type")) not in {"compositional", "inference"}:
            continue
        evidences = row.get("evidences") or []
        if len(evidences) != 2 or len(evidences[-1]) < 3:
            continue
        subject, relation, obj = [str(x).strip() for x in evidences[-1][:3]]
        answer = str(row.get("answer") or "").strip()
        complex_q = str(row.get("question") or "").strip()
        template = RELATION_TEMPLATES.get(relation)
        if not template or not subject or not answer or not complex_q:
            continue
        if not answer_matches(obj, answer):
            continue
        simple_q = template.format(subject=subject)
        if normalize(simple_q) == normalize(complex_q):
            continue
        candidates.append({
            "pair_id": str(row.get("_id")),
            "source_type": str(row.get("type")),
            "relation": relation,
            "subject": subject,
            "gold_answer": answer,
            "simple_question": simple_q,
            "complex_question": complex_q,
            "evidences": evidences,
        })

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_pairs]


def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> Tuple[str, int]:
    prompt = FEW_SHOT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    first_token = int(new_tokens[0].item()) if len(new_tokens) else int(tokenizer.eos_token_id)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    for stop in ["\nQ:", "\n\n", "\n"]:
        if stop in text:
            text = text.split(stop)[0]
            break
    return text.strip(), first_token


def evidence_question(pair: dict, difficulty: str) -> str:
    triples = pair["evidences"][-1:] if difficulty == "simple" else pair["evidences"]
    facts = "\n".join(
        f"- {str(s).strip()} has relation '{str(r).strip()}' with {str(o).strip()}."
        for s, r, o in triples
    )
    base = pair[f"{difficulty}_question"]
    return f"Use the provided facts to answer.\nFacts:\n{facts}\nQuestion: {base}"


def generate_factorial_rows(
    assets,
    pairs: List[dict],
    max_new_tokens: int,
    with_evidence: bool = False,
) -> List[dict]:
    rows = []
    for pair in tqdm(pairs, desc="v35-generate-pairs"):
        for difficulty in ["simple", "complex"]:
            question = (
                evidence_question(pair, difficulty)
                if with_evidence
                else pair[f"{difficulty}_question"]
            )
            answer, first_token = generate_answer(
                assets.model, assets.tokenizer, question, max_new_tokens
            )
            correct = is_answer_correct(answer, [pair["gold_answer"]])
            rows.append({
                **pair,
                "difficulty": difficulty,
                "question": question,
                "model_answer": answer,
                "model_correct": bool(correct),
                "first_token_id": first_token,
            })
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_rows() -> List[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def confidence_features(logits: torch.Tensor, token_id: int) -> np.ndarray:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum()
    return np.asarray([
        float(log_probs[token_id].detach().cpu()),
        float(probs.max().detach().cpu()),
        float(entropy.detach().cpu()),
    ], dtype=np.float32)


def extract_states(assets, rows: List[dict]) -> dict:
    hooks = {
        layer: ResidualHook().attach(assets.model.model.layers[layer])
        for layer in LAYERS
    }
    sae = {stage: {layer: [] for layer in LAYERS} for stage in ["T0", "T1"]}
    raw = {stage: {layer: [] for layer in LAYERS} for stage in ["T0", "T1"]}
    conf = {stage: [] for stage in ["T0", "T1"]}
    try:
        for row in tqdm(rows, desc="v35-extract-states"):
            prompt = FEW_SHOT_TEMPLATE.format(question=row["question"])
            inputs = assets.tokenizer(prompt, return_tensors="pt").to(assets.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out0 = assets.model(**inputs, use_cache=False)
            conf["T0"].append(confidence_features(
                out0.logits[0, -1], int(row["first_token_id"])
            ))
            for layer in LAYERS:
                vec = hooks[layer].value[0, prompt_len - 1]
                raw["T0"][layer].append(vec.float().cpu().numpy())
                with torch.no_grad():
                    z = assets.saes[layer].encode(vec.unsqueeze(0)).squeeze(0)
                sae["T0"][layer].append(z.float().cpu().numpy())

            first = torch.tensor([[int(row["first_token_id"])]], device=assets.device)
            t1_ids = torch.cat([inputs["input_ids"], first], dim=1)
            t1_mask = torch.ones_like(t1_ids)
            with torch.no_grad():
                out1 = assets.model(input_ids=t1_ids, attention_mask=t1_mask, use_cache=False)
            conf["T1"].append(confidence_features(
                out1.logits[0, -1], int(out1.logits[0, -1].argmax().item())
            ))
            for layer in LAYERS:
                vec = hooks[layer].value[0, prompt_len]
                raw["T1"][layer].append(vec.float().cpu().numpy())
                with torch.no_grad():
                    z = assets.saes[layer].encode(vec.unsqueeze(0)).squeeze(0)
                sae["T1"][layer].append(z.float().cpu().numpy())
    finally:
        for hook in hooks.values():
            hook.detach()

    payload = {
        "labels": np.asarray([0 if r["model_correct"] else 1 for r in rows], dtype=np.int64),
        "difficulty": np.asarray([r["difficulty"] for r in rows], dtype=object),
        "pair_ids": np.asarray([r["pair_id"] for r in rows], dtype=object),
    }
    for stage in ["T0", "T1"]:
        payload[f"conf_{stage}"] = np.stack(conf[stage])
        for layer in LAYERS:
            payload[f"sae_{stage}_L{layer}"] = np.stack(sae[stage][layer])
            payload[f"raw_{stage}_L{layer}"] = np.stack(raw[stage][layer])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **payload)
    return payload


def load_states() -> dict:
    arr = np.load(CACHE_PATH, allow_pickle=True)
    return {key: arr[key] for key in arr.files}


def make_clf(kind: str, seed: int) -> LogisticRegression:
    if kind in {"sae", "fused"}:
        return LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            class_weight="balanced", max_iter=3000, random_state=seed,
        )
    return LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0,
        class_weight="balanced", max_iter=3000, random_state=seed,
    )


def fit_predict(X_train, y_train, X_test, kind: str, seed: int) -> np.ndarray:
    sparse = kind in {"sae", "fused"}
    scaler = StandardScaler(with_mean=not sparse)
    Xtr = scaler.fit_transform(X_train.astype(np.float64))
    Xte = scaler.transform(X_test.astype(np.float64))
    clf = make_clf(kind, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"auroc": None, "auprc": None, "n": int(len(y))}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "n": int(len(y)),
    }


def grouped_evaluation(X: np.ndarray, y: np.ndarray, difficulty: np.ndarray, groups: np.ndarray, kind: str) -> List[dict]:
    rows = []
    for seed in SEEDS:
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(seed)
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        fold_map = {g: i % 5 for i, g in enumerate(shuffled)}
        fold_probs = {
            "pooled": np.full(len(y), np.nan),
            "simple_to_complex": np.full(len(y), np.nan),
            "complex_to_simple": np.full(len(y), np.nan),
        }
        for fold in range(5):
            test = np.asarray([fold_map[g] == fold for g in groups])
            train = ~test
            fold_probs["pooled"][test] = fit_predict(
                X[train], y[train], X[test], kind, seed + fold
            )
            for source, target, name in [
                ("simple", "complex", "simple_to_complex"),
                ("complex", "simple", "complex_to_simple"),
            ]:
                tr = train & (difficulty == source)
                te = test & (difficulty == target)
                if len(np.unique(y[tr])) < 2 or not np.any(te):
                    continue
                fold_probs[name][te] = fit_predict(
                    X[tr], y[tr], X[te], kind, seed + fold
                )
        for protocol, probs in fold_probs.items():
            valid = np.isfinite(probs)
            metrics = safe_metrics(y[valid], probs[valid])
            rows.append({"seed": seed, "protocol": protocol, **metrics})
    return rows


def summarize_eval(rows: List[dict]) -> List[dict]:
    out = []
    for protocol in sorted({r["protocol"] for r in rows}):
        selected = [r for r in rows if r["protocol"] == protocol and r["auroc"] is not None]
        if not selected:
            continue
        out.append({
            "protocol": protocol,
            "n_seeds": len(selected),
            "auroc_mean": float(np.mean([r["auroc"] for r in selected])),
            "auroc_std": float(np.std([r["auroc"] for r in selected], ddof=1)),
            "auprc_mean": float(np.mean([r["auprc"] for r in selected])),
            "auprc_std": float(np.std([r["auprc"] for r in selected], ddof=1)),
            "n": selected[0]["n"],
        })
    return out


def invariant_effects(X: np.ndarray, y: np.ndarray, difficulty: np.ndarray) -> dict:
    effects = {}
    for level in ["simple", "complex"]:
        mask = difficulty == level
        wrong = X[mask & (y == 1)]
        correct = X[mask & (y == 0)]
        effects[level] = wrong.mean(axis=0) - correct.mean(axis=0)
    pooled_std = X.std(axis=0) + 1e-6
    ds = effects["simple"] / pooled_std
    dc = effects["complex"] / pooled_std
    same_sign = np.sign(ds) == np.sign(dc)
    substantive = (np.abs(ds) >= 0.10) & (np.abs(dc) >= 0.10)
    selected = np.where(same_sign & substantive)[0]
    scores = np.minimum(np.abs(ds[selected]), np.abs(dc[selected]))
    order = selected[np.argsort(-scores)] if len(selected) else selected
    return {
        "same_sign_fraction": float(np.mean(same_sign)),
        "shared_effect_count_abs_ge_0_10": int(len(selected)),
        "top_shared_indices": [int(x) for x in order[:30]],
        "top_shared_simple_effect": [float(ds[x]) for x in order[:30]],
        "top_shared_complex_effect": [float(dc[x]) for x in order[:30]],
        "effect_cosine": float(np.dot(ds, dc) / (np.linalg.norm(ds) * np.linalg.norm(dc) + 1e-12)),
    }


def main() -> None:
    global OUT_PATH, DATA_PATH, CACHE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--with-evidence", action="store_true")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    if args.with_evidence:
        DATA_PATH = CONTEXT_DATA_PATH
        CACHE_PATH = CONTEXT_CACHE_PATH
        if args.out == str(OUT_PATH):
            args.out = str(CONTEXT_OUT_PATH)

    assets = None
    if args.force_generate or not DATA_PATH.exists():
        pairs = build_pairs(args.n_pairs, args.seed)
        if len(pairs) < args.n_pairs:
            raise RuntimeError(f"Only found {len(pairs)} valid pairs")
        assets = load_all(args.config)
        rows = generate_factorial_rows(
            assets,
            pairs,
            args.max_new_tokens,
            with_evidence=args.with_evidence,
        )
    else:
        rows = load_rows()
        print(f"[v35] loaded rows -> {DATA_PATH}")

    if args.force_extract or not CACHE_PATH.exists():
        if assets is None:
            assets = load_all(args.config)
        states = extract_states(assets, rows)
    else:
        states = load_states()
        print(f"[v35] loaded states -> {CACHE_PATH}")

    y = states["labels"].astype(int)
    difficulty = states["difficulty"].astype(str)
    groups = states["pair_ids"].astype(str)
    cell_counts = {}
    for level in ["simple", "complex"]:
        for outcome, label in [("correct", 0), ("wrong", 1)]:
            cell_counts[f"{level}_{outcome}"] = int(np.sum((difficulty == level) & (y == label)))

    eval_rows = []
    summaries = []
    effects = []
    for stage in ["T0", "T1"]:
        conf_X = states[f"conf_{stage}"].astype(np.float64)
        rows_conf = grouped_evaluation(conf_X, y, difficulty, groups, "conf")
        for row in rows_conf:
            eval_rows.append({"stage": stage, "layer": None, "method": "confidence", **row})
        for row in summarize_eval(rows_conf):
            summaries.append({"stage": stage, "layer": None, "method": "confidence", **row})

        for layer in LAYERS:
            sae_X = states[f"sae_{stage}_L{layer}"].astype(np.float64)
            raw_X = states[f"raw_{stage}_L{layer}"].astype(np.float64)
            fused_X = np.hstack([sae_X, conf_X])
            for method, X, kind in [
                ("sae", sae_X, "sae"),
                ("raw_residual", raw_X, "raw"),
                ("sae_conf_fused", fused_X, "fused"),
            ]:
                method_rows = grouped_evaluation(X, y, difficulty, groups, kind)
                for row in method_rows:
                    eval_rows.append({"stage": stage, "layer": layer, "method": method, **row})
                for row in summarize_eval(method_rows):
                    summaries.append({"stage": stage, "layer": layer, "method": method, **row})
            effects.append({
                "stage": stage,
                "layer": layer,
                **invariant_effects(sae_X, y, difficulty),
            })

    summaries.sort(key=lambda r: (r["auroc_mean"] if r["auroc_mean"] is not None else -1), reverse=True)
    payload = {
        "experiment": "poc_v35_factorial_pilot",
        "dataset": "2Wiki matched simple/complex",
        "with_evidence": bool(args.with_evidence),
        "n_pairs": int(len(np.unique(groups))),
        "n_rows": int(len(y)),
        "cell_counts": cell_counts,
        "simple_accuracy": float(1.0 - y[difficulty == "simple"].mean()),
        "complex_accuracy": float(1.0 - y[difficulty == "complex"].mean()),
        "layers": LAYERS,
        "stages": ["T0", "T1"],
        "data_path": str(DATA_PATH),
        "cache_path": str(CACHE_PATH),
        "summary": summaries,
        "effect_overlap": effects,
        "rows": eval_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[v35] cells={cell_counts}")
    print(f"[v35] simple_acc={payload['simple_accuracy']:.3f} complex_acc={payload['complex_accuracy']:.3f}")
    print(f"[v35] saved -> {out}")
    for row in summaries[:20]:
        print(
            f"[v35] {row['protocol']} {row['stage']} {row['method']} L{row['layer']} "
            f"AUROC={row['auroc_mean']:.4f}±{row['auroc_std']:.4f} "
            f"AUPRC={row['auprc_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
