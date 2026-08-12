"""V46: isolate the nonlinear value of the CEVR risk adapter."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_cevr_cross_dataset_router import load_npz  # noqa: E402
from scripts.poc_v44_framework_guided_cevr_finetuning import (  # noqa: E402
    SEEDS,
    UniqueCandidates,
    collapse_candidates,
    mixed_question_mask,
    unique_choice,
    within_question_auroc,
)


N_BOOTSTRAP = 10_000


class AffineUtility(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class RankUtility(nn.Module):
    def __init__(self, dimension: int, rank: int, nonlinear: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, 1)
        self.down = nn.Linear(dimension, rank, bias=False)
        self.out = nn.Linear(rank, 1)
        self.dropout = nn.Dropout(0.10)
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down(x)
        if self.nonlinear:
            hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        return (self.linear(x) + self.out(hidden)).squeeze(-1)


def make_model(name: str, dimension: int) -> nn.Module:
    if name == "affine":
        return AffineUtility(dimension)
    if name == "deep_linear":
        return RankUtility(dimension, rank=32, nonlinear=False)
    if name == "gelu":
        return RankUtility(dimension, rank=32, nonlinear=True)
    raise ValueError(name)


def train_model(
    name: str,
    x: np.ndarray,
    correct: np.ndarray,
    qids: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int = 100,
) -> nn.Module:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(name, x.shape[1]).to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(correct, dtype=torch.bool, device=device)
    _, group_numbers = np.unique(qids, return_inverse=True)
    group_ids = torch.as_tensor(group_numbers, dtype=torch.long, device=device)
    n_groups = int(group_ids.max().item()) + 1
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        utility = model(xt)
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
        correct_sum.scatter_add_(0, group_ids[yt], utility[yt])
        correct_count.scatter_add_(0, group_ids[yt], torch.ones_like(utility[yt]))
        if not bool(torch.all(correct_count == 1)):
            raise ValueError("Expected exactly one correct unique candidate per group")
        loss = (log_normalizer - correct_sum).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def metrics(data: UniqueCandidates, utility: np.ndarray) -> dict:
    selected_correct = []
    selected_by_question = {}
    for qid in np.unique(data.question_ids):
        indices = np.flatnonzero(data.question_ids == qid)
        chosen = indices[np.argmax(utility[indices])]
        value = int(data.correct[chosen])
        selected_correct.append(value)
        selected_by_question[str(qid)] = value
    return {
        "question_accuracy": float(np.mean(selected_correct)),
        "candidate_auroc": float(roc_auc_score(data.correct, utility)),
        "candidate_auprc": float(average_precision_score(data.correct, utility)),
        "within_question_auroc": within_question_auroc(
            data.correct, data.question_ids, utility
        ),
        "selected_correct_by_question": selected_by_question,
    }


def paired_delta(
    reference: dict[str, int], challenger: dict[str, int], seed: int
) -> dict:
    qids = sorted(set(reference) & set(challenger))
    delta = np.asarray([challenger[qid] - reference[qid] for qid in qids])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(N_BOOTSTRAP, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
        "n_questions": int(len(delta)),
    }


def subset(data: UniqueCandidates, mask: np.ndarray) -> UniqueCandidates:
    return UniqueCandidates(**{
        field: getattr(data, field)[mask]
        for field in data.__dataclass_fields__
    })


def oof_models(
    train: UniqueCandidates, device: torch.device
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    mask = mixed_question_mask(train)
    data = subset(train, mask)
    splitter = GroupKFold(n_splits=5)
    names = ("affine", "deep_linear", "gelu")
    scores = {name: np.zeros(len(data.correct), dtype=np.float64) for name in names}
    for train_index, valid_index in splitter.split(
        data.x, data.correct, data.question_ids
    ):
        scaler = StandardScaler()
        train_z = scaler.fit_transform(data.x[train_index]).astype(np.float32)
        valid_z = scaler.transform(data.x[valid_index]).astype(np.float32)
        for name in names:
            seed_scores = []
            for seed in SEEDS:
                model = train_model(
                    name, train_z, data.correct[train_index],
                    data.question_ids[train_index], seed, device,
                )
                seed_scores.append(predict(model, valid_z, device))
            scores[name][valid_index] = np.mean(seed_scores, axis=0)
    return {name: metrics(data, values) for name, values in scores.items()}, scores


def external_models(
    train: UniqueCandidates,
    target: UniqueCandidates,
    device: torch.device,
) -> dict[str, dict]:
    mask = mixed_question_mask(train)
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train.x[mask]).astype(np.float32)
    target_z = scaler.transform(target.x).astype(np.float32)
    results = {}
    for name in ("affine", "deep_linear", "gelu"):
        seed_scores = []
        seed_choices = []
        parameter_count = None
        for seed in SEEDS:
            model = train_model(
                name, train_z, train.correct[mask], train.question_ids[mask],
                seed, device,
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            values = predict(model, target_z, device)
            seed_scores.append(values)
            seed_choices.append(unique_choice(target, values))
        utility = np.mean(seed_scores, axis=0)
        result = metrics(target, utility)
        result["parameter_count"] = int(parameter_count)
        result["three_seed_unanimous_choice_fraction"] = float(np.mean([
            len({choice[qid] for choice in seed_choices}) == 1
            for qid in sorted(seed_choices[0])
        ]))
        results[name] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="outputs/poc_v46_nonlinear_adapter_attribution_results.json"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cache = load_npz(ROOT / "outputs/cache/v40f_no_rationale_states.npz")
    train = collapse_candidates(
        train_cache,
        ROOT / "outputs/poc_v40b_commitment_generation_results.json",
        "raw_L18", clean=True,
    )
    oof, _ = oof_models(train, device)
    print(json.dumps({name: {key: value for key, value in row.items()
                            if key != "selected_correct_by_question"}
                      for name, row in oof.items()}, indent=2), flush=True)

    oof_deltas = {
        "gelu_minus_affine": paired_delta(
            oof["affine"]["selected_correct_by_question"],
            oof["gelu"]["selected_correct_by_question"], 20260930,
        ),
        "gelu_minus_deep_linear": paired_delta(
            oof["deep_linear"]["selected_correct_by_question"],
            oof["gelu"]["selected_correct_by_question"], 20260931,
        ),
    }

    targets = []
    target_specs = (
        ("V41", "outputs/cache/v41_hotpot_no_rationale_states.npz",
         "outputs/poc_v41_hotpot_no_rationale_generation_results.json"),
        ("V42", "outputs/cache/v42_hotpot_no_rationale_states.npz",
         "outputs/poc_v42_hotpot_no_rationale_generation_results.json"),
        ("V45", "outputs/cache/v45_hotpot_no_rationale_states.npz",
         "outputs/poc_v45_hotpot_no_rationale_generation_results.json"),
    )
    pooled = {name: {} for name in ("affine", "deep_linear", "gelu")}
    for dataset, cache_path, generation_path in target_specs:
        cache = load_npz(ROOT / cache_path)
        target = collapse_candidates(
            cache, ROOT / generation_path, "raw_C_L18", clean=False
        )
        result = external_models(train, target, device)
        deltas = {
            "gelu_minus_affine": paired_delta(
                result["affine"]["selected_correct_by_question"],
                result["gelu"]["selected_correct_by_question"], 20260940,
            ),
            "gelu_minus_deep_linear": paired_delta(
                result["deep_linear"]["selected_correct_by_question"],
                result["gelu"]["selected_correct_by_question"], 20260941,
            ),
        }
        for name in pooled:
            pooled[name].update({
                f"{dataset}::{qid}": value
                for qid, value in result[name]["selected_correct_by_question"].items()
            })
        targets.append({"dataset": dataset, "models": result, "paired": deltas})
        print(json.dumps({
            "dataset": dataset,
            "accuracy": {name: row["question_accuracy"] for name, row in result.items()},
            "paired": deltas,
        }, indent=2), flush=True)

    pooled_delta = paired_delta(pooled["deep_linear"], pooled["gelu"], 20260950)
    gate_1 = (
        oof["gelu"]["question_accuracy"] >= oof["deep_linear"]["question_accuracy"]
        and oof["gelu"]["within_question_auroc"]
        > oof["deep_linear"]["within_question_auroc"]
    )
    gate_2 = pooled_delta["ci95"][0] > 0.0
    payload = {
        "experiment": "poc_v46_nonlinear_adapter_attribution",
        "protocol": "paper/V46_NONLINEAR_ADAPTER_ATTRIBUTION_PROTOCOL.md",
        "device": str(device),
        "v40_grouped_oof": oof,
        "v40_paired": oof_deltas,
        "historical_targets": targets,
        "pooled_historical_gelu_minus_deep_linear": pooled_delta,
        "backbone_lora_gate": {
            "oof_noninferior_accuracy_and_higher_within_auroc": bool(gate_1),
            "pooled_historical_ci_strictly_positive": bool(gate_2),
            "overall_pass": bool(gate_1 and gate_2),
        },
        "claim_boundary": [
            "V41, V42, and V45 are historical replay sets for V46.",
            "The matched comparison isolates GELU from parameter count but remains a monitor-readout experiment.",
            "A positive gate requires a new prospective LoRA protocol before any backbone-training claim.",
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "oof_paired": oof_deltas,
        "pooled": pooled_delta,
        "gate": payload["backbone_lora_gate"],
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
