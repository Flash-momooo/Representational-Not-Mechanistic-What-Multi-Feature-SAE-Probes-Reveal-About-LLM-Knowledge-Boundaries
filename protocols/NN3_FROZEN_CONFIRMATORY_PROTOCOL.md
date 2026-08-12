# NN3 Frozen Sparse-Readout Confirmation Protocol

Protocol freeze date: 2026-07-26. This file is written before NN3 question
selection, generation, state extraction, or label inspection.

## Question

Does validation-selected dynamic residual sparsity preserve the predictive and
calibration quality of a full dense residual readout on previously untouched
questions and in two model families?

## Frozen Data

- TriviaQA `rc.nocontext` validation split from the local official parquet.
- Exclude every normalized prompt present in any existing project JSONL.
- Deduplicate by normalized prompt and source question ID.
- Select 200 questions with seed 20260727: 100 WikipediaEntity answers and 100
  other answer types.
- No question may be replaced after any model output is observed.
- Eight trajectories per question, temperature 0.7, top-p 0.9, top-k 50, and
  at most 16 new tokens.

## Frozen Models

1. `google/gemma-2-2b`, base shortest-answer prompt, layer 18, generation seed
   20260922. GemmaScope 16K latents are encoded for a secondary comparison.
2. `Qwen/Qwen2.5-1.5B-Instruct`, chat template, normalized-depth layer 19,
   generation seed 20260921.

## Frozen Algorithm

- Five pair-ID-grouped outer folds with seed 20260726.
- Endpoint: first-divergence next-token hazard at T1--T3.
- Candidate capacities: K in {1, 2, 4, 8, 16, 32, 64}.
- In each outer fold, `(fold + 1) mod 5` is a capacity-selection fold. The
  remaining three folds fit stage-specific supports for each K.
- Select one global K per outer fold by minimum question-grouped NLL pooled
  over T1--T3. Then refit stage-specific supports on all four outer-training
  folds and evaluate once on the untouched outer test fold.
- All scaling, support selection, PCA, and classifier fitting occur inside the
  relevant training split.
- Primary sparse method: adaptive supervised raw-residual coordinates plus the
  ten inherited confidence statistics.
- Secondary methods: dense residual, confidence, fixed raw K=8, PCA-8, L1-8,
  Gaussian RP-8, 20 random K=8 supports, and adaptive SAE coordinates for
  Gemma only.

## Frozen Endpoints And Criteria

Primary trajectory score is the maximum available OOF T1--T3 hazard
probability. Report AUROC, AUPRC, Brier score, question-grouped paired
bootstrap intervals, and strictly pre-event recall under the inherited
calibration protocol.

The adaptive raw readout passes compression retention in a model when:

1. the 95% interval for AUROC(adaptive - dense) has lower endpoint at least
   -0.02; and
2. the 95% interval for Brier(adaptive - dense) has upper endpoint at most
   +0.01.

Passing in both models is the confirmatory criterion. Superiority, SAE
specificity, and exact 10% FPR control are secondary and are not required for
confirmation. A failed criterion is retained and cannot trigger changes to K,
folds, features, questions, or labels.
