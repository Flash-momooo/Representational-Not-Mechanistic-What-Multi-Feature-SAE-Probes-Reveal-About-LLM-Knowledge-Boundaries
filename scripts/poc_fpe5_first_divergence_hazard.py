"""FPE5: machine-verifiable first-divergence hazard and latency audit."""

from __future__ import annotations

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

try:
    from poc_fpe3_trajectory_dynamics import (
        SEEDS,
        bootstrap,
        fit_scalers,
        question_weighted_differences,
        score_metrics,
        solve_energy,
        transform,
    )
except ModuleNotFoundError:
    from scripts.poc_fpe3_trajectory_dynamics import (
        SEEDS,
        bootstrap,
        fit_scalers,
        question_weighted_differences,
        score_metrics,
        solve_energy,
        transform,
    )


ROOT = Path(__file__).resolve().parents[1]
MAX_STAGE = 5
METHOD_ORDER = (
    "confidence",
    "token_prefix",
    "position_l18",
    "velocity_l18",
    "sae_velocity_l18",
    "velocity_plus_token",
    "sae_velocity_plus_token",
)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prefix_status(text: str, gold: str, eos: bool = False) -> str:
    """Return risk, event, or success for the first nonempty generated line."""
    first_line = None
    line_terminated = False
    for part in text.splitlines(keepends=True):
        content = part.rstrip("\r\n").strip()
        if content:
            first_line = content
            line_terminated = part.endswith(("\n", "\r"))
            break
    if first_line is None:
        candidate = ""
    else:
        candidate = re.sub(
            r"^answer\s*:\s*", "", first_line, flags=re.IGNORECASE
        ).strip()
    candidate = normalize_answer(candidate)
    if candidate and not gold.startswith(candidate):
        return "event"
    if line_terminated or eos:
        return "success" if candidate == gold else "event"
    return "risk"


def load_event_table(
    tokenizer, data_path: Path, cache_path: Path
) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    rows = [
        json.loads(line)
        for line in data_path
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    cache_file = np.load(cache_path, allow_pickle=True)
    states = {key: cache_file[key] for key in cache_file.files}
    n = len(rows)
    risk = np.zeros((MAX_STAGE + 1, n), dtype=bool)
    event = np.zeros((MAX_STAGE + 1, n), dtype=np.int8)
    success = np.zeros((MAX_STAGE + 1, n), dtype=np.int8)
    token_transition = np.zeros((MAX_STAGE + 1, n), dtype=bool)
    final_status = []

    for row_index, row in enumerate(rows):
        token_ids = [int(value) for value in row["generated_token_ids"]]
        gold = normalize_answer(row["gold_answer"])
        terminal = "risk"
        for stage in range(MAX_STAGE + 1):
            if stage > len(token_ids):
                continue
            before = prefix_status(
                tokenizer.decode(token_ids[:stage], skip_special_tokens=True), gold
            )
            if before != "risk":
                continue
            risk[stage, row_index] = True
            if stage < len(token_ids):
                after = prefix_status(
                    tokenizer.decode(token_ids[: stage + 1], skip_special_tokens=True),
                    gold,
                )
                token_transition[stage, row_index] = True
            else:
                after = prefix_status(
                    tokenizer.decode(token_ids, skip_special_tokens=True), gold, eos=True
                )
            event[stage, row_index] = int(after == "event")
            success[stage, row_index] = int(after == "success")

        for stage in range(1, len(token_ids) + 1):
            terminal = prefix_status(
                tokenizer.decode(token_ids[:stage], skip_special_tokens=True), gold
            )
            if terminal != "risk":
                break
        if terminal == "risk":
            terminal = prefix_status(
                tokenizer.decode(token_ids, skip_special_tokens=True), gold, eos=True
            )
        final_status.append(terminal)

    original_correct = np.asarray([row["model_correct"] for row in rows], dtype=bool)
    event_correct = np.asarray([value == "success" for value in final_status])
    agreement = {
        "n_rows": n,
        "n_agree": int(np.sum(original_correct == event_correct)),
        "agreement_rate": float(np.mean(original_correct == event_correct)),
        "original_correct": int(original_correct.sum()),
        "event_process_success": int(event_correct.sum()),
        "false_event_under_original_label": int(np.sum(original_correct & ~event_correct)),
        "false_success_under_original_label": int(np.sum(~original_correct & event_correct)),
    }
    table = {"risk": risk, "event": event, "success": success,
             "token_transition": token_transition}
    return rows, states, {"table": table, "agreement": agreement}


def split_pair_folds(pair_ids: np.ndarray, seed: int):
    unique = np.unique(pair_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {pair_id: index % 5 for index, pair_id in enumerate(unique)}
    for fold in range(5):
        yield (
            np.asarray([mapping[pair_id] != fold for pair_id in pair_ids]),
            np.asarray([mapping[pair_id] == fold for pair_id in pair_ids]),
        )


def methods_for_stage(stage: int) -> tuple[str, ...]:
    if stage == 0:
        return METHOD_ORDER[:3]
    return METHOD_ORDER


def blocks(states: dict[str, np.ndarray], stage: int, method: str):
    confidence = states[f"confidence_T{stage}"].astype(np.float32)
    token = states[f"token_prefix_T{stage}"].astype(np.float32)
    position = states[f"raw_T{stage}_L18"].astype(np.float32)
    if method == "confidence":
        return [confidence]
    if method == "token_prefix":
        return [token]
    if method == "position_l18":
        return [position]
    dense_velocity = position - states[f"raw_T{stage - 1}_L18"].astype(np.float32)
    sae_velocity = (
        states[f"sae_T{stage}_L18"].astype(np.float32)
        - states[f"sae_T{stage - 1}_L18"].astype(np.float32)
    )
    if method == "velocity_l18":
        return [dense_velocity]
    if method == "sae_velocity_l18":
        return [sae_velocity]
    if method == "velocity_plus_token":
        return [dense_velocity, token]
    if method == "sae_velocity_plus_token":
        return [sae_velocity, token]
    raise ValueError(method)


def evaluate(
    states: dict[str, np.ndarray],
    feature_stage: int,
    labels: np.ndarray,
    valid: np.ndarray,
    device: str,
) -> tuple[list[dict], dict, list[dict]]:
    question_ids = states["question_ids"].astype(str)
    pair_ids = states["pair_ids"].astype(str)
    result_rows = []
    local_store = {}
    diagnostics = []
    for seed in SEEDS:
        predictions = {
            method: np.full(len(labels), np.nan, dtype=np.float32)
            for method in methods_for_stage(feature_stage)
        }
        for fold, (train_rows, test_rows) in enumerate(split_pair_folds(pair_ids, seed)):
            train_valid = valid & train_rows
            for method in methods_for_stage(feature_stage):
                feature_blocks = blocks(states, feature_stage, method)
                scalers = fit_scalers(feature_blocks, train_valid)
                features = transform(feature_blocks, scalers)
                differences, weights, n_questions = question_weighted_differences(
                    features, labels, question_ids, train_valid
                )
                if n_questions == 0:
                    continue
                direction, info = solve_energy(differences, weights, device)
                selected = valid & test_rows
                predictions[method][selected] = features[selected] @ direction
                diagnostics.append({
                    "feature_stage": f"T{feature_stage}", "method": method,
                    "seed": seed, "fold": fold,
                    "n_discordant_train_questions": n_questions, **info,
                })
        for method, scores in predictions.items():
            metric, local = score_metrics(labels, scores, question_ids, valid)
            local_store[(method, seed)] = local
            result_rows.append({
                "feature_stage": f"T{feature_stage}", "method": method,
                "seed": seed, **metric,
            })
    return result_rows, local_store, diagnostics


def summarize(result_rows: list[dict], analysis: str, transition: int) -> list[dict]:
    output = []
    methods = sorted({row["method"] for row in result_rows})
    for method in methods:
        selected = [row for row in result_rows if row["method"] == method]
        output.append({
            "analysis": analysis,
            "transition": f"T{transition}->T{transition + 1}",
            "feature_stage": selected[0]["feature_stage"],
            "method": method,
            "within_question_auroc_macro_mean": float(np.mean([
                row["within_question_auroc_macro"] for row in selected
            ])),
            "population_auroc_mean": float(np.mean([
                row["population_auroc"] for row in selected
            ])),
            "population_auprc_mean": float(np.mean([
                row["population_auprc"] for row in selected
            ])),
            "n_rows": selected[0]["n_rows"],
            "n_discordant_questions": selected[0]["n_discordant_questions"],
        })
    return output


def exact_prefix_audit(rows, states, table, tokenizer) -> list[dict]:
    question_ids = states["question_ids"].astype(str)
    audits = []
    for stage in range(MAX_STAGE + 1):
        valid = table["risk"][stage]
        labels = table["event"][stage]
        groups = defaultdict(list)
        for row_index in np.flatnonzero(valid):
            prefix = tuple(rows[row_index]["generated_token_ids"][:stage])
            groups[(question_ids[row_index], prefix)].append(row_index)
        discordant = 0
        tied_pairs = 0
        all_pairs = 0
        oracle_scores = np.zeros(len(rows), dtype=np.float64)
        max_variance = 0.0
        for indices in groups.values():
            local = labels[indices]
            rate = float(local.mean())
            oracle_scores[indices] = rate
            positives = int(local.sum())
            negatives = len(local) - positives
            tied_pairs += positives * negatives
            if positives and negatives:
                discordant += 1
            raw = states[f"raw_T{stage}_L18"][indices].astype(np.float32)
            max_variance = max(max_variance, float(np.max(np.var(raw, axis=0))))
        for question in np.unique(question_ids[valid]):
            local = valid & (question_ids == question)
            positives = int(labels[local].sum())
            all_pairs += positives * (int(local.sum()) - positives)
        metric, _ = score_metrics(labels, oracle_scores, question_ids, valid)
        audits.append({
            "stage": f"T{stage}", "n_risk_rows": int(valid.sum()),
            "n_events": int(labels[valid].sum()), "n_exact_prefix_groups": len(groups),
            "n_discordant_exact_prefix_groups": discordant,
            "max_raw_l18_variance_within_exact_prefix_group": max_variance,
            "within_question_positive_negative_pairs": all_pairs,
            "exact_prefix_tied_pair_fraction": (
                float(tied_pairs / all_pairs) if all_pairs else float("nan")
            ),
            "empirical_exact_prefix_oracle_within_question_auroc":
                metric["within_question_auroc_macro"],
        })
    return audits


def paired_bootstrap(local_left: dict, local_right: dict, seed: int) -> dict:
    common = sorted(set(local_left) & set(local_right))
    if not common:
        return {
            "mean": None, "ci95": [None, None], "reference": 0.0,
            "probability_at_or_below_reference": None, "n_questions": 0,
        }
    differences = np.asarray(
        [local_left[q] - local_right[q] for q in common], dtype=np.float64
    )
    return bootstrap(differences, seed, 0.0)


def averaged_seed_bootstrap(
    local_results: dict,
    left_keys: list[tuple],
    right_keys: list[tuple],
    seed: int,
) -> dict:
    common = sorted(set.intersection(*[
        set(local_results[key]) for key in left_keys + right_keys
    ]))
    if not common:
        return {
            "mean": None, "ci95": [None, None], "reference": 0.0,
            "probability_at_or_below_reference": None, "n_questions": 0,
        }
    differences = np.asarray([
        np.mean([
            local_results[left][question] - local_results[right][question]
            for left, right in zip(left_keys, right_keys)
        ])
        for question in common
    ], dtype=np.float64)
    return bootstrap(differences, seed, 0.0)


def run(
    data_path: Path | None = None,
    cache_path: Path | None = None,
    output_path: Path | None = None,
    config_path: Path | None = None,
    experiment_suffix: str = "",
) -> dict:
    data_path = data_path or ROOT / "data" / "v37_2wiki_trajectory.jsonl"
    cache_path = cache_path or ROOT / "outputs" / "cache" / "v37_2wiki_trajectory_states.npz"
    output_path = output_path or ROOT / "outputs" / "poc_fpe5_first_divergence_hazard_results.json"
    config_path = config_path or ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], local_files_only=True
    )
    rows, states, event_data = load_event_table(tokenizer, data_path, cache_path)
    table = event_data["table"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []
    all_diagnostics = []
    local_results = {}

    for transition in range(MAX_STAGE + 1):
        valid = table["risk"][transition]
        labels = table["event"][transition]
        result, local, diagnostics = evaluate(
            states, transition, labels, valid, device
        )
        all_rows.extend(summarize(result, "pre_transition_hazard", transition))
        all_diagnostics.extend(diagnostics)
        for key, value in local.items():
            local_results[("pre", transition, *key)] = value

        if transition < MAX_STAGE:
            paired_valid = valid & table["token_transition"][transition]
            _, pre_paired_local, diagnostics = evaluate(
                states, transition, labels, paired_valid, device
            )
            all_diagnostics.extend(diagnostics)
            for key, value in pre_paired_local.items():
                local_results[("pre_paired", transition, *key)] = value
            result, local, diagnostics = evaluate(
                states, transition + 1, labels, paired_valid, device
            )
            all_rows.extend(summarize(result, "post_transition_detection", transition))
            all_diagnostics.extend(diagnostics)
            for key, value in local.items():
                local_results[("post", transition, *key)] = value

    comparisons = []
    for transition in range(MAX_STAGE):
        shared_methods = set(methods_for_stage(transition)) & set(
            methods_for_stage(transition + 1)
        )
        for method in sorted(shared_methods):
            comparisons.append({
                "contrast": f"post minus pre {method} at T{transition}->T{transition + 1}",
                "seeds_averaged": list(SEEDS),
                "bootstrap": averaged_seed_bootstrap(
                    local_results,
                    [("post", transition, method, seed) for seed in SEEDS],
                    [("pre_paired", transition, method, seed) for seed in SEEDS],
                    2026071500 + 20 * transition,
                ),
            })

    for transition in range(1, MAX_STAGE + 1):
        for left, right in (
            ("position_l18", "token_prefix"),
            ("velocity_l18", "token_prefix"),
            ("sae_velocity_l18", "token_prefix"),
            ("sae_velocity_l18", "velocity_l18"),
            ("sae_velocity_plus_token", "token_prefix"),
            ("velocity_plus_token", "token_prefix"),
        ):
            comparisons.append({
                "contrast": f"pre {left} minus {right} at T{transition}",
                "seeds_averaged": list(SEEDS),
                "bootstrap": averaged_seed_bootstrap(
                    local_results,
                    [("pre", transition, left, seed) for seed in SEEDS],
                    [("pre", transition, right, seed) for seed in SEEDS],
                    2026071800 + 20 * transition,
                ),
            })

    payload = {
        "experiment": "FPE5 first-divergence discrete hazard and latency" + experiment_suffix,
        "model": cfg["model"]["name"],
        "data_path": str(data_path),
        "cache_path": str(cache_path),
        "event_definition": "first irreversible normalized exact-answer divergence",
        "label_agreement": event_data["agreement"],
        "identifiability_audit": exact_prefix_audit(rows, states, table, tokenizer),
        "summary": all_rows,
        "comparisons": comparisons,
        "local_aurocs": {
            "|".join(map(str, key)): value for key, value in local_results.items()
        },
        "fit_diagnostics": all_diagnostics,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {output_path}")
    print(json.dumps(payload["label_agreement"], indent=2))
    print(json.dumps(payload["identifiability_audit"], indent=2))
    for row in all_rows:
        print(
            f"{row['analysis']:<25} {row['transition']} {row['method']:<24} "
            f"within={row['within_question_auroc_macro_mean']:.4f} "
            f"n={row['n_rows']} q={row['n_discordant_questions']}"
        )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v37_2wiki_trajectory.jsonl")
    parser.add_argument("--cache", default="outputs/cache/v37_2wiki_trajectory_states.npz")
    parser.add_argument("--output", default="outputs/poc_fpe5_first_divergence_hazard_results.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    run(
        ROOT / args.data, ROOT / args.cache, ROOT / args.output,
        ROOT / args.config, args.suffix,
    )
