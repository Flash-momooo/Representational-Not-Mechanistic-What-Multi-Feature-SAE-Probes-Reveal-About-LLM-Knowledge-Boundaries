# NN4 Frozen 7B Model-Scale Extension Protocol

Protocol freeze date: 2026-07-26. Written before Qwen2.5-7B outputs are
generated on the NN3 questions.

## Purpose

Test whether the frozen NN3 adaptive sparse-readout algorithm retains a full
dense residual readout when model scale increases from 1.5B/2B to 7B.

## Frozen Setup

- Questions: the unchanged 200 zero-overlap NN3 TriviaQA questions.
- Model: local `Qwen2.5-7B-Instruct` checkpoint.
- Loading: bitsandbytes NF4, double quantization, bfloat16 computation.
- Prompt: chat template and shortest-answer instruction.
- Layer: normalized-depth rule `round((18/26) * n_layers)`.
- Eight trajectories per question, temperature 0.7, top-p 0.9, top-k 50,
  maximum 16 new tokens, generation seed 20260923.
- No SAE is available for this model; only raw-residual methods are evaluated.

## Frozen Analysis

Use the NN3 algorithm and implementation without changes:

- five pair-ID-grouped outer folds, seed 20260726;
- K in {1, 2, 4, 8, 16, 32, 64};
- rotating validation fold selects one global K by question-grouped T1--T3 NLL;
- stage-specific supports are refit on the four outer-training folds;
- primary trajectory score is maximum OOF T1--T3 hazard probability;
- 2,000 question-grouped paired bootstrap repetitions.

The extension passes when the 95% interval for AUROC(adaptive - dense) has
lower endpoint at least -0.02 and the interval for Brier(adaptive - dense) has
upper endpoint at most +0.01. Results are explicitly limited to fixed NF4
loading and are not treated as a full-precision invariance result.
