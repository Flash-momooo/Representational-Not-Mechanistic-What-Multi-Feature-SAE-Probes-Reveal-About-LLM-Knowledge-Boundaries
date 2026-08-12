# NN32 Frozen Qwen-7B True-SAE Replication

Freeze date: 2026-07-30.

## Purpose

Test whether the conditional-compressibility result observed with the
GemmaScope SAE can be assessed with a *real* SAE on a second model family.
This is not a claim that the resulting small SAE has the coverage or feature
quality of GemmaScope. It is a strictly isolated scale-and-family replication
of the sparse-readout comparison.

## Frozen Assets and Isolation

- Backbone: local `Qwen2.5-7B-Instruct`, NF4 double quantization with bfloat16
  computation, matching FPE20.
- SAE layer: residual stream output of transformer layer 18, the same hook and
  tensor convention as the frozen NN19 Qwen cache.
- SAE training corpus: all 240 prompts in
  `data/fpe12_trivia_holdout_questions.jsonl`. Only prompt-token residuals are
  used. Gold answers, model generations, correctness labels, and WebQuestions
  content are never read by the SAE trainer.
- SAE validation: the final 20% of those 240 training prompts, split at the
  prompt level before any activation is collected.
- Test set: the untouched 600-question WebQuestions trajectories and residual
  cache fixed by NN19. No SAE hyperparameter, layer, expansion, or optimizer
  setting may be changed after their results are inspected.

## Dictionary and Readout

- Dictionary: a two-times-overcomplete Top-K SAE (`d_model=3584`,
  `d_sae=7168`, `K=32`), trained for three epochs with MSE reconstruction.
- Standardization: a training-corpus mean is subtracted before encoding and
  added after decoding.
- Quality audit: report held-out reconstruction MSE, explained variance,
  cosine similarity, and mean L0. A failed audit remains reportable; it
  invalidates any positive SAE-specific conclusion.
- Risk readouts: confidence, dense residual, raw-coordinate Top-K, and SAE
  Top-K. In each outer question fold, the candidate width is selected only on
  a disjoint selection fold. The untouched outer test fold reports AUROC,
  AUPRC, Brier score, support stability, and nominal-10%-FPR pre-event utility.

## Interpretation Rule

The experiment can establish only whether this independently trained Qwen SAE
matches or materially trails dense/raw readouts under the stated checkpoint,
corpus, and small-dictionary budget. It cannot establish SAE superiority,
feature semantics, full-precision invariance, or a universal cross-model SAE
law.

## Dataset-Size Correction

The protocol was checked immediately before collection. The referenced holdout file contains 240, rather than 400, question prompts. No model forward pass or SAE update had occurred at that point. Therefore the frozen corpus is all 240 prompts, with the first 192 for training and final 48 for validation; all other settings above are unchanged.
