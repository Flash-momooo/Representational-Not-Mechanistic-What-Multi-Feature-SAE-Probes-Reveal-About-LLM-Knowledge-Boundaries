# FPE12 Frozen TriviaQA Capability-Boundary Protocol

Protocol freeze date: 2026-07-17. Frozen before selecting questions and before
Qwen2.5-1.5B generation on those questions.

## Motivation

FPE10 and FPE11 PopQA targets are far above the tested models' capability
boundary. Most wrong trajectories leave the exact gold-prefix set at the first
token, leaving only four T1-discordant questions in each Qwen condition. This is
a support limitation of the hazard endpoint, not evidence for or against a
trajectory monitor.

FPE12 uses a zero-overlap slice of the established TriviaQA `rc.nocontext`
validation split. PopQA remains the frozen high-difficulty stress test; FPE12 is
the predeclared capability-boundary confirmation.

## Frozen Data

- Local official TriviaQA validation parquet: 17,944 questions.
- Exclude every normalized prompt present in any existing local TriviaQA JSONL.
- Sample 240 questions with seed 20260717, balanced 120/120 between answer type
  `WikipediaEntity` and all other answer types. If a stratum has fewer than 120
  eligible rows, report preparation failure rather than changing strata.
- Model, layer, prompt, generation parameters, and K are identical to FPE11.
- Eight trajectories per question, generation seed 20260803.
- Source remains the already frozen FPE11 Qwen2.5-1.5B 2Wiki source. No FPE12
  output may alter source selection or model fitting rules.

## Two Frozen Endpoints

1. **Next-token hazard:** among trajectories still compatible with a gold answer
   prefix at T1, predict whether the next token first leaves that prefix set.
   This supports a strict pre-event claim but can have low support.
2. **Early final outcome:** among all trajectories with at least one generated
   token, predict final answer error from the T1 residual delta and confidence.
   This measures early information and intervention utility, but errors already
   committed at the first token are not called pre-event predictions.

Both endpoints report global and within-question metrics. Neither substitutes
for the other.

## Frozen Methods and Criteria

- Methods: confidence-only, dense residual, raw top-K (K=8), PCA-8, Gaussian
  RP-8, L1 raw-8, and 20 random-coordinate supports.
- Primary candidate: inherited raw top-K.
- At least 30 discordant target questions are required for a strong
  within-question endpoint claim.
- Compression retention: raw-8 AUROC at least dense AUROC minus 0.03.
- Internal increment: raw-8 AUROC at least confidence AUROC plus 0.02, with a
  question-grouped 95% bootstrap interval not wholly negative.
- Random supports are summarized as a distribution; the best random seed is not
  selected.

Any failure remains in the result. No question, layer, K, read position, or
endpoint definition may be changed after generation.

