"""V44: framework-guided fine-tuning of a small CEVR risk adapter.

The language model remains frozen.  This experiment fine-tunes a compact
nonlinear readout on post-commitment, candidate-relative residual states and
uses a question-listwise objective.  Same-capacity controls separate the
effect of capacity from the framework's stage, representation, and target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import (  # noqa: E402
    accuracy_from_indices,
    bootstrap_delta,
    candidate_metrics,
    center_by_question,
    choose_per_question,
    load_npz,
    risk_scores,
)
from scripts.poc_v40_extract_and_evaluate import clean_rows, load_generation  # noqa: E402


SEEDS = (20260808, 20260809, 20260810)
RANKS = (8, 16, 32)


@dataclass
class UniqueCandidates:
    x: np.ndarray
    correct: np.ndarray
    question_ids: np.ndarray
    options: np.ndarray
    likelihood: np.ndarray
    sample_indices: np.ndarray
    source_indices: np.ndarray


class RiskAdapter(nn.Module):
    """Linear utility plus a low-rank nonlinear correction."""

    def __init__(self, dimension: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, 1)
        self.down = nn.Linear(dimension, rank, bias=False)
        self.out = nn.Linear(rank, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.out(self.dropout(F.gelu(self.down(x))))
        return (self.linear(x) + correction).squeeze(-1)


def generation_lookup(path: Path) -> dict[tuple[str, int], dict]:
    rows = load_generation(path)
    return {
        (str(row["item_id"]), int(row["sample_index"])): row
        for row in rows
    }


def row_metadata(
    cache: dict[str, np.ndarray], generation_path: Path, clean: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qids = cache["question_ids"].astype(str)
    if clean:
        rows = clean_rows(load_generation(generation_path))
        row_qids = np.asarray([str(row["item_id"]) for row in rows])
        if not np.array_equal(qids, row_qids):
            raise ValueError("Clean generation rows no longer align with training cache")
        options = np.asarray([str(row["selected_option"]) for row in rows])
        sample_indices = np.asarray([int(row["sample_index"]) for row in rows])
        likelihood = np.asarray([
            float(row["candidate_mean_logprob"][row["selected_option"]])
            for row in rows
        ], dtype=np.float32)
        return options, sample_indices, likelihood

    options = cache["selected_options"].astype(str)
    sample_indices = cache["sample_indices"].astype(int)
    lookup = generation_lookup(generation_path)
    likelihood = []
    for qid, sample_index, option in zip(qids, sample_indices, options):
        row = lookup[(str(qid), int(sample_index))]
        if str(row["selected_option"]) != option:
            raise ValueError(f"Option mismatch for {(qid, sample_index)}")
        likelihood.append(float(row["candidate_mean_logprob"][option]))
    return options, sample_indices, np.asarray(likelihood, dtype=np.float32)


def collapse_candidates(
    cache: dict[str, np.ndarray],
    generation_path: Path,
    feature_key: str,
    clean: bool,
) -> UniqueCandidates:
    qids = cache["question_ids"].astype(str)
    options, sample_indices, likelihood = row_metadata(cache, generation_path, clean)
    labels = cache["labels"].astype(int)
    features = cache[feature_key].astype(np.float32)
    output = {key: [] for key in (
        "x", "correct", "question_ids", "options", "likelihood",
        "sample_indices", "source_indices",
    )}
    seen: set[tuple[str, str]] = set()
    order = np.lexsort((sample_indices, qids))
    for index in order:
        key = (str(qids[index]), str(options[index]))
        if key in seen:
            continue
        seen.add(key)
        members = np.flatnonzero((qids == key[0]) & (options == key[1]))
        candidate_labels = labels[members]
        if len(np.unique(candidate_labels)) != 1:
            raise ValueError(f"Inconsistent labels for candidate {key}")
        first = members[np.argmin(sample_indices[members])]
        output["x"].append(features[members].mean(axis=0))
        output["correct"].append(1 - int(candidate_labels[0]))
        output["question_ids"].append(key[0])
        output["options"].append(key[1])
        output["likelihood"].append(float(likelihood[members].mean()))
        output["sample_indices"].append(int(sample_indices[first]))
        output["source_indices"].append(int(first))
    return UniqueCandidates(
        x=np.asarray(output["x"], dtype=np.float32),
        correct=np.asarray(output["correct"], dtype=np.int64),
        question_ids=np.asarray(output["question_ids"], dtype=str),
        options=np.asarray(output["options"], dtype=str),
        likelihood=np.asarray(output["likelihood"], dtype=np.float32),
        sample_indices=np.asarray(output["sample_indices"], dtype=np.int64),
        source_indices=np.asarray(output["source_indices"], dtype=np.int64),
    )


def mixed_question_mask(data: UniqueCandidates) -> np.ndarray:
    valid_questions = []
    for qid in np.unique(data.question_ids):
        y = data.correct[data.question_ids == qid]
        if len(np.unique(y)) == 2:
            valid_questions.append(qid)
    return np.isin(data.question_ids, valid_questions)


def question_groups(question_ids: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(question_ids == qid) for qid in np.unique(question_ids)]


def prepare_features(
    train_x: np.ndarray,
    train_qids: np.ndarray,
    other_x: np.ndarray,
    other_qids: np.ndarray,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    if centered:
        train_x = center_by_question(train_x, train_qids)
        other_x = center_by_question(other_x, other_qids)
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x).astype(np.float32)
    other_z = scaler.transform(other_x).astype(np.float32)
    return train_z, other_z, scaler


def train_adapter(
    x: np.ndarray,
    correct: np.ndarray,
    question_ids: np.ndarray,
    rank: int,
    objective: str,
    seed: int,
    epochs: int,
    device: torch.device,
) -> RiskAdapter:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = RiskAdapter(x.shape[1], rank, dropout=0.10).to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(correct, dtype=torch.float32, device=device)
    _, group_numbers = np.unique(question_ids, return_inverse=True)
    group_ids = torch.as_tensor(group_numbers, dtype=torch.long, device=device)
    n_groups = int(group_ids.max().item()) + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    positives = max(float(correct.sum()), 1.0)
    negatives = max(float(len(correct) - correct.sum()), 1.0)
    pos_weight = torch.tensor(negatives / positives, device=device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        utility = model(xt)
        if objective == "bce":
            loss = F.binary_cross_entropy_with_logits(
                utility, yt, pos_weight=pos_weight
            )
        elif objective == "listwise":
            maxima = torch.full(
                (n_groups,), -torch.inf, dtype=utility.dtype, device=device
            )
            maxima.scatter_reduce_(
                0, group_ids, utility.detach(), reduce="amax", include_self=True
            )
            exp_sum = torch.zeros(n_groups, dtype=utility.dtype, device=device)
            exp_sum.scatter_add_(0, group_ids, torch.exp(utility - maxima[group_ids]))
            log_normalizer = maxima + torch.log(exp_sum)
            correct_sum = torch.zeros(n_groups, dtype=utility.dtype, device=device)
            correct_count = torch.zeros(n_groups, dtype=utility.dtype, device=device)
            correct_mask = yt.bool()
            correct_sum.scatter_add_(
                0, group_ids[correct_mask], utility[correct_mask]
            )
            correct_count.scatter_add_(
                0, group_ids[correct_mask], torch.ones_like(utility[correct_mask])
            )
            if not bool(torch.all(correct_count == 1)):
                raise ValueError("Each collapsed mixed question must have one correct option")
            loss = (log_normalizer - correct_sum).mean()
        else:
            raise ValueError(objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model


@torch.inference_mode()
def predict(model: RiskAdapter, x: np.ndarray, device: torch.device) -> np.ndarray:
    return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def selection_accuracy(
    correct: np.ndarray, question_ids: np.ndarray, utility: np.ndarray
) -> float:
    values = []
    for qid in np.unique(question_ids):
        indices = np.flatnonzero(question_ids == qid)
        chosen = indices[np.argmax(utility[indices])]
        values.append(int(correct[chosen]))
    return float(np.mean(values))


def within_question_auroc(
    correct: np.ndarray, question_ids: np.ndarray, utility: np.ndarray
) -> float:
    values = []
    for qid in np.unique(question_ids):
        indices = np.flatnonzero(question_ids == qid)
        if len(np.unique(correct[indices])) == 2:
            values.append(roc_auc_score(correct[indices], utility[indices]))
    return float(np.mean(values))


def grouped_oof(
    data: UniqueCandidates,
    rank: int,
    centered: bool,
    objective: str,
    epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    mask = mixed_question_mask(data)
    x = data.x[mask]
    y = data.correct[mask]
    qids = data.question_ids[mask]
    splitter = GroupKFold(n_splits=5)
    predictions = np.zeros(len(y), dtype=np.float64)
    for train_index, valid_index in splitter.split(x, y, qids):
        train_z, valid_z, _ = prepare_features(
            x[train_index], qids[train_index], x[valid_index], qids[valid_index], centered
        )
        fold_scores = []
        for seed in SEEDS:
            model = train_adapter(
                train_z, y[train_index], qids[train_index], rank,
                objective, seed, epochs, device,
            )
            fold_scores.append(predict(model, valid_z, device))
        predictions[valid_index] = np.mean(fold_scores, axis=0)
    metrics = {
        "question_accuracy": selection_accuracy(y, qids, predictions),
        "candidate_auroc": float(roc_auc_score(y, predictions)),
        "candidate_auprc": float(average_precision_score(y, predictions)),
        "within_question_auroc": within_question_auroc(y, qids, predictions),
        "n_rows": int(len(y)),
        "n_questions": int(len(np.unique(qids))),
    }
    return predictions, metrics


def fit_ensemble(
    train: UniqueCandidates,
    target: UniqueCandidates,
    rank: int,
    centered: bool,
    objective: str,
    epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray], int, StandardScaler]:
    mask = mixed_question_mask(train)
    train_z, target_z, scaler = prepare_features(
        train.x[mask], train.question_ids[mask], target.x, target.question_ids, centered
    )
    predictions = []
    parameter_count = 0
    for seed in SEEDS:
        model = train_adapter(
            train_z, train.correct[mask], train.question_ids[mask], rank,
            objective, seed, epochs, device,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        predictions.append(predict(model, target_z, device))
    return np.mean(predictions, axis=0), predictions, parameter_count, scaler


def unique_choice(data: UniqueCandidates, utility: np.ndarray) -> dict[str, int]:
    selected = {}
    for qid in np.unique(data.question_ids):
        indices = np.flatnonzero(data.question_ids == qid)
        order = np.lexsort((data.sample_indices[indices], -utility[indices]))
        selected[str(qid)] = int(data.source_indices[indices[order[0]]])
    return selected


def first_choice(cache: dict[str, np.ndarray]) -> dict[str, int]:
    qids = cache["question_ids"].astype(str)
    sample_indices = cache["sample_indices"].astype(int)
    return {
        str(qid): int(np.flatnonzero((qids == qid) & (sample_indices == 0))[0])
        for qid in np.unique(qids)
    }


def evaluate_target(
    name: str,
    train_cache: dict[str, np.ndarray],
    target_cache: dict[str, np.ndarray],
    train: UniqueCandidates,
    target: UniqueCandidates,
    target_generation: Path,
    selected_rank: int,
    epochs: int,
    device: torch.device,
) -> dict:
    labels = target_cache["labels"].astype(int)
    qids = target_cache["question_ids"].astype(str)
    sample_indices = target_cache["sample_indices"].astype(int)
    first = first_choice(target_cache)
    likelihood = unique_choice(target, target.likelihood)

    original_risk = risk_scores(
        train_cache, target_cache, "raw_L18", "raw_C_L18"
    )
    original_linear = choose_per_question(qids, sample_indices, -original_risk)

    train_mask = mixed_question_mask(train)
    train_center = center_by_question(train.x[train_mask], train.question_ids[train_mask])
    target_center = center_by_question(target.x, target.question_ids)
    linear_scaler = StandardScaler()
    linear = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=4000,
        random_state=20260808,
    )
    linear.fit(linear_scaler.fit_transform(train_center), train.correct[train_mask])
    unique_linear_utility = linear.predict_proba(linear_scaler.transform(target_center))[:, 1]
    unique_linear = unique_choice(target, unique_linear_utility)

    methods = {}
    specifications = (
        ("absolute_listwise_adapter", False, "listwise"),
        ("centered_bce_adapter", True, "bce"),
        ("centered_listwise_adapter", True, "listwise"),
    )
    full_scaler = None
    for method_name, centered, objective in specifications:
        utility, seed_predictions, parameter_count, scaler = fit_ensemble(
            train, target, selected_rank, centered, objective, epochs, device
        )
        if method_name == "centered_listwise_adapter":
            full_scaler = scaler
        choice = unique_choice(target, utility)
        seed_choices = [unique_choice(target, values) for values in seed_predictions]
        question_order = sorted(choice)
        agreement = np.mean([
            len({seed_choice[qid] for seed_choice in seed_choices}) == 1
            for qid in question_order
        ])
        result = accuracy_from_indices(labels, choice)
        result.update({
            "paired_vs_first": bootstrap_delta(labels, first, choice, 20260880),
            "paired_vs_likelihood": bootstrap_delta(labels, likelihood, choice, 20260881),
            "paired_vs_original_linear": bootstrap_delta(
                labels, original_linear, choice, 20260882
            ),
            "candidate_level_unique": {
                "population_auroc": float(roc_auc_score(1 - target.correct, -utility)),
                "population_auprc": float(average_precision_score(1 - target.correct, -utility)),
                "within_question_auroc": within_question_auroc(
                    target.correct, target.question_ids, utility
                ),
            },
            "parameter_count": int(parameter_count),
            "three_seed_unanimous_choice_fraction": float(agreement),
        })
        methods[method_name] = result

    baseline_choices = {
        "first_sample": first,
        "restricted_likelihood": likelihood,
        "original_linear_cevr": original_linear,
        "unique_centered_logistic": unique_linear,
    }
    baselines = {}
    for baseline_name, choice in baseline_choices.items():
        result = accuracy_from_indices(labels, choice)
        if baseline_name != "first_sample":
            result["paired_vs_first"] = bootstrap_delta(
                labels, first, choice, 20260890
            )
        if baseline_name == "unique_centered_logistic":
            result["paired_vs_likelihood"] = bootstrap_delta(
                labels, likelihood, choice, 20260891
            )
            result["paired_vs_original_linear"] = bootstrap_delta(
                labels, original_linear, choice, 20260892
            )
        baselines[baseline_name] = result

    # Same-capacity T0 control: within a question all pre-sampling states are
    # identical, so candidate centering removes every candidate distinction.
    t0 = target_cache["raw_T0_L18"].astype(np.float32)
    max_collision_error = 0.0
    for qid in np.unique(qids):
        indices = np.flatnonzero(qids == qid)
        max_collision_error = max(
            max_collision_error,
            float(np.max(np.abs(t0[indices] - t0[indices[0]]))),
        )
    t0_centered = center_by_question(t0, qids)
    max_centered_magnitude = float(np.max(np.abs(t0_centered)))
    t0_choice = first
    t0_result = accuracy_from_indices(labels, t0_choice)
    t0_result.update({
        "within_question_auroc": 0.5,
        "max_within_question_state_difference": max_collision_error,
        "max_centered_state_magnitude": max_centered_magnitude,
        "decision": "all candidate scores tied; earliest sample selected",
    })

    return {
        "dataset": name,
        "n_rows": int(len(labels)),
        "n_questions": int(len(np.unique(qids))),
        "n_unique_sampled_candidates": int(len(target.correct)),
        "n_mixed_questions": int(len(np.unique(
            target.question_ids[mixed_question_mask(target)]
        ))),
        "baselines": baselines,
        "adapters": methods,
        "t0_collision_control": t0_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="outputs/cache/v40f_no_rationale_states.npz")
    parser.add_argument("--train-generation", default="outputs/poc_v40b_commitment_generation_results.json")
    parser.add_argument("--v41-cache", default="outputs/cache/v41_hotpot_no_rationale_states.npz")
    parser.add_argument("--v41-generation", default="outputs/poc_v41_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--v42-cache", default="outputs/cache/v42_hotpot_no_rationale_states.npz")
    parser.add_argument("--v42-generation", default="outputs/poc_v42_hotpot_no_rationale_generation_results.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="outputs/poc_v44_framework_guided_cevr_finetuning_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = load_npz(ROOT / args.train_cache)
    train = collapse_candidates(
        train_cache, ROOT / args.train_generation, "raw_L18", clean=True
    )

    rank_results = []
    for rank in RANKS:
        _, metrics = grouped_oof(
            train, rank, centered=True, objective="listwise",
            epochs=args.epochs, device=device,
        )
        rank_results.append({"rank": rank, **metrics})
        print(f"rank={rank} oof={metrics}", flush=True)
    selected = sorted(
        rank_results,
        key=lambda row: (-row["question_accuracy"], -row["within_question_auroc"], row["rank"]),
    )[0]
    selected_rank = int(selected["rank"])

    control_oof = {}
    for method_name, centered, objective in (
        ("absolute_listwise_adapter", False, "listwise"),
        ("centered_bce_adapter", True, "bce"),
        ("centered_listwise_adapter", True, "listwise"),
    ):
        _, metrics = grouped_oof(
            train, selected_rank, centered=centered, objective=objective,
            epochs=args.epochs, device=device,
        )
        control_oof[method_name] = metrics
        print(f"{method_name} oof={metrics}", flush=True)

    targets = []
    for name, cache_path, generation_path in (
        ("V41 / HotpotQA retrospective", args.v41_cache, args.v41_generation),
        ("V42 / HotpotQA source-isolated prospective set", args.v42_cache, args.v42_generation),
    ):
        target_cache = load_npz(ROOT / cache_path)
        target = collapse_candidates(
            target_cache, ROOT / generation_path, "raw_C_L18", clean=False
        )
        result = evaluate_target(
            name, train_cache, target_cache, train, target,
            ROOT / generation_path, selected_rank, args.epochs, device,
        )
        targets.append(result)
        print(json.dumps({
            "dataset": name,
            "baseline_accuracy": {
                key: value["accuracy"] for key, value in result["baselines"].items()
            },
            "adapter_accuracy": {
                key: value["accuracy"] for key, value in result["adapters"].items()
            },
            "t0": result["t0_collision_control"],
        }, indent=2), flush=True)

    full_oof = control_oof["centered_listwise_adapter"]
    capacity_control = control_oof["absolute_listwise_adapter"]
    objective_control = control_oof["centered_bce_adapter"]
    gate_1 = (
        full_oof["question_accuracy"] > capacity_control["question_accuracy"]
        and full_oof["question_accuracy"] > objective_control["question_accuracy"]
    )
    gate_2 = all(
        target["adapters"]["centered_listwise_adapter"]["accuracy"]
        >= target["baselines"]["original_linear_cevr"]["accuracy"]
        for target in targets
    )
    gate_3 = all(
        math.isclose(target["t0_collision_control"]["within_question_auroc"], 0.5)
        and math.isclose(
            target["t0_collision_control"]["accuracy"],
            target["baselines"]["first_sample"]["accuracy"],
        )
        for target in targets
    )
    payload = {
        "experiment": "poc_v44_framework_guided_cevr_finetuning",
        "protocol": "paper/V44_FRAMEWORK_GUIDED_CEVR_FINETUNING_PROTOCOL.md",
        "status": "exploratory framework-guided adapter test; targets are historical external benchmarks",
        "device": str(device),
        "train": {
            "dataset": "V40f / 2WikiMultiHopQA",
            "n_original_rows": int(len(train_cache["labels"])),
            "n_unique_sampled_candidates": int(len(train.correct)),
            "n_questions": int(len(np.unique(train.question_ids))),
            "n_mixed_questions": int(len(np.unique(
                train.question_ids[mixed_question_mask(train)]
            ))),
        },
        "rank_selection": rank_results,
        "selected_rank": selected_rank,
        "v40_grouped_oof_controls": control_oof,
        "targets": targets,
        "pre_specified_gate": {
            "framework_method_beats_both_same_capacity_oof_controls": bool(gate_1),
            "framework_method_not_below_original_linear_on_both_targets": bool(gate_2),
            "t0_collision_control_passes": bool(gate_3),
            "overall_pass": bool(gate_1 and gate_2 and gate_3),
        },
        "claim_boundary": [
            "The base language model is frozen; V44 fine-tunes a compact residual-state risk adapter.",
            "V41 and V42 are not untouched confirmations for V44 because earlier CEVR results were inspected.",
            "A positive configuration requires one further prospectively isolated dataset before a confirmatory claim.",
            "T0 results concern within-question future-branch identification, not question-level difficulty prediction.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "selected_rank": selected_rank,
        "gate": payload["pre_specified_gate"],
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
