"""V55: output-only multi-sample semantic entropy and SelfCheck-style audit.

This uses eight *actual sampled* answers per question.  It deliberately does
not treat fixed machine-verifiable answer options as samples.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fpe5c3b_gemma2_it_hotpot_trajectory.jsonl"
OUTPUT = ROOT / "outputs" / "poc_v55_multisample_semantic_baselines.json"
JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_BOOTSTRAP = 10_000
SEED = 20260828
JUDGE_SYSTEM = (
    "You are a strict semantic-equivalence verifier. Reply Yes only when the two "
    "answers assert the same answer to the question. Reply No when they name "
    "different entities, values, relations, one answer refuses, or either answer "
    "is unsupported. Do not infer that different answers are equivalent."
)


def canonical(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def question_from_prompt(prompt: str) -> str:
    marker = "Question:"
    if marker not in prompt:
        return ""
    value = prompt.split(marker, 1)[1]
    return value.split("\nAnswer:", 1)[0].strip()


def judge_prompt(question: str, left: str, right: str) -> str:
    return (
        "Are these two answers semantically equivalent? Reply only Yes or No.\n\n"
        f"Question: {question}\nAnswer 1: {left}\nAnswer 2: {right}\nEquivalent?"
    )


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def entropy_from_components(components: list[list[int]]) -> float:
    masses = np.asarray([len(component) for component in components], dtype=float)
    masses /= masses.sum()
    return float(-(masses * np.log(masses)).sum())


def component_lists(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    union_find = UnionFind(n)
    for left, right in edges:
        union_find.union(left, right)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(n):
        components[union_find.find(index)].append(index)
    return list(components.values())


def yes_probabilities(prompts: list[str], batch_size: int, device: str, model_id: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "7B" in str(model_id):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, device_map="auto",
            quantization_config=quant, torch_dtype=torch.bfloat16,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode("No", add_special_tokens=False)[-1]
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            rendered = [tokenizer.apply_chat_template(
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            ) for prompt in prompts[start:start + batch_size]]
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True,
                truncation=True, max_length=768,
            ).to(device)
            logits = model(**encoded).logits[:, -1, [no_id, yes_id]].float()
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            print(f"judged {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float32)


def macro_within_question_auc(labels: np.ndarray, scores: np.ndarray, groups: list[list[int]]) -> float | None:
    values = []
    for group in groups:
        y = labels[group]
        if np.unique(y).size == 2:
            values.append(float(roc_auc_score(y, scores[group])))
    return float(np.mean(values)) if values else None


def choice_from_components(components: list[list[int]], pair_yes: np.ndarray) -> int:
    # Largest semantic component; use mean internal support then sample order to resolve ties.
    ranked = sorted(components, key=lambda component: (-len(component), min(component)))
    largest_size = len(ranked[0])
    contenders = [component for component in ranked if len(component) == largest_size]
    if len(contenders) == 1:
        return min(contenders[0])
    support = []
    for component in contenders:
        values = [pair_yes[i, j] for i in component for j in component if i != j]
        support.append((float(np.mean(values)) if values else 1.0, -min(component), component))
    return min(max(support, key=lambda item: (item[0], item[1]))[2])


def bootstrap_summary(question_records: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(question_records)
    arrays = {
        key: np.asarray([float(record[key]) for record in question_records])
        for key in ("first_correct", "lexical_mode_correct", "semantic_mode_correct", "selfcheck_correct", "oracle_correct")
    }
    distribution = {key: np.empty(N_BOOTSTRAP, dtype=float) for key in arrays}
    distribution["semantic_minus_first"] = np.empty(N_BOOTSTRAP, dtype=float)
    distribution["selfcheck_minus_first"] = np.empty(N_BOOTSTRAP, dtype=float)
    for bootstrap_index in range(N_BOOTSTRAP):
        sampled = rng.integers(0, n, size=n)
        for key, value in arrays.items():
            distribution[key][bootstrap_index] = value[sampled].mean()
        distribution["semantic_minus_first"][bootstrap_index] = (
            arrays["semantic_mode_correct"][sampled] - arrays["first_correct"][sampled]
        ).mean()
        distribution["selfcheck_minus_first"][bootstrap_index] = (
            arrays["selfcheck_correct"][sampled] - arrays["first_correct"][sampled]
        ).mean()
    result = {}
    for key, value in distribution.items():
        result[key] = {
            "estimate": float(arrays[key].mean()) if key in arrays else float(
                (arrays["semantic_mode_correct"] - arrays["first_correct"]).mean()
                if key == "semantic_minus_first"
                else (arrays["selfcheck_correct"] - arrays["first_correct"]).mean()
            ),
            "ci95": [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    items = [sorted(grouped[key], key=lambda row: int(row["sample_index"])) for key in sorted(grouped)[:args.n_questions]]
    if not items or any(len(item) != 8 for item in items):
        raise ValueError("V55 requires exactly eight real sampled answers per question.")

    prompts, pair_metadata = [], []
    for question_index, item in enumerate(items):
        question = question_from_prompt(item[0]["prompt"])
        for left in range(8):
            for right in range(left + 1, 8):
                # Exact canonical equality is a deterministic semantic-equivalence
                # anchor. Only non-identical strings are delegated to the small judge.
                if canonical(item[left]["model_answer"]) != canonical(item[right]["model_answer"]):
                    prompts.append(judge_prompt(question, item[left]["model_answer"], item[right]["model_answer"]))
                    pair_metadata.append((question_index, left, right))
    scores = yes_probabilities(prompts, args.batch_size, args.device, args.judge_model)

    pair_scores = [np.eye(8, dtype=np.float32) for _ in items]
    for question_index, item in enumerate(items):
        for left in range(8):
            for right in range(left + 1, 8):
                if canonical(item[left]["model_answer"]) == canonical(item[right]["model_answer"]):
                    pair_scores[question_index][left, right] = 1.0
                    pair_scores[question_index][right, left] = 1.0
    for score, (question_index, left, right) in zip(scores, pair_metadata):
        pair_scores[question_index][left, right] = score
        pair_scores[question_index][right, left] = score

    labels, selfcheck_risk, semantic_entropy = [], [], []
    lexical_entropy, question_records = [], []
    groups = [list(range(index * 8, (index + 1) * 8)) for index in range(len(items))]
    for question_index, item in enumerate(items):
        answers = [row["model_answer"] for row in item]
        correct = np.asarray([bool(row["model_correct"]) for row in item], dtype=bool)
        canonical_answers = [canonical(answer) for answer in answers]
        lexical_groups: dict[str, list[int]] = defaultdict(list)
        for answer_index, answer in enumerate(canonical_answers):
            lexical_groups[answer].append(answer_index)
        lex_components = list(lexical_groups.values())
        matrix = pair_scores[question_index]
        semantic_components = component_lists(8, [(left, right) for left in range(8) for right in range(left + 1, 8) if matrix[left, right] >= 0.5])
        semantic_choice = choice_from_components(semantic_components, matrix)
        lexical_choice = min(max(lex_components, key=lambda component: (len(component), -min(component))))
        selfcheck_choice = int(np.argmin(1.0 - (matrix.sum(axis=1) - 1.0) / 7.0))
        branch_risk = 1.0 - (matrix.sum(axis=1) - 1.0) / 7.0
        labels.extend((~correct).astype(np.int8).tolist())
        selfcheck_risk.extend(branch_risk.tolist())
        semantic_entropy.extend([entropy_from_components(semantic_components)] * 8)
        lexical_entropy.extend([entropy_from_components(lex_components)] * 8)
        question_records.append({
            "question_id": item[0]["question_id"],
            "first_correct": int(correct[0]),
            "lexical_mode_correct": int(correct[lexical_choice]),
            "semantic_mode_correct": int(correct[semantic_choice]),
            "selfcheck_correct": int(correct[selfcheck_choice]),
            "oracle_correct": int(correct.any()),
            "semantic_entropy": entropy_from_components(semantic_components),
            "lexical_entropy": entropy_from_components(lex_components),
        })

    labels_np = np.asarray(labels, dtype=np.int8)
    selfcheck_np = np.asarray(selfcheck_risk, dtype=float)
    semantic_np = np.asarray(semantic_entropy, dtype=float)
    lexical_np = np.asarray(lexical_entropy, dtype=float)
    first_labels = np.asarray([1 - record["first_correct"] for record in question_records], dtype=np.int8)
    semantic_question = np.asarray([record["semantic_entropy"] for record in question_records], dtype=float)
    lexical_question = np.asarray([record["lexical_entropy"] for record in question_records], dtype=float)

    output = {
        "experiment": "V55 real-trajectory semantic entropy and SelfCheck-style audit",
        "frozen_scope": {
            "trajectory_file": str(DATA.relative_to(ROOT)), "questions": len(items),
            "samples_per_question": 8, "completions": len(labels),
            "judge": str(args.judge_model), "judge_mode": "local deterministic Yes/No pairwise semantic equivalence",
            "pair_prompts": len(prompts), "target_note": "gold labels are evaluation-only",
        },
        "semantic_cluster_rule": "union pair scores >= 0.5; entropy over component mass",
        "question_level_first_sample_error_auroc": {
            "semantic_entropy": float(roc_auc_score(first_labels, semantic_question)),
            "lexical_agreement_entropy_control": float(roc_auc_score(first_labels, lexical_question)),
        },
        "branch_level_macro_within_question_auroc": {
            "selfcheck_style_semantic_disagreement": macro_within_question_auc(labels_np, selfcheck_np, groups),
            "semantic_entropy_question_score": macro_within_question_auc(labels_np, semantic_np, groups),
            "lexical_entropy_question_score": macro_within_question_auc(labels_np, lexical_np, groups),
        },
        "selection_bootstrap": bootstrap_summary(question_records),
        "mean_semantic_entropy": float(semantic_question.mean()),
        "mean_lexical_entropy": float(lexical_question.mean()),
        "interpretation_limits": [
            "Semantic entropy is a canonical-equality-anchored, Qwen-judged clustering approximation, not a verbatim reproduction of every Semantic Entropy implementation.",
            "SelfCheck-style disagreement is inspired by SelfCheckGPT but uses a frozen local equivalence judge, not its original evaluator suite.",
            "This is retrospective on an existing 80-question trajectory collection and has additional pairwise judge cost.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
