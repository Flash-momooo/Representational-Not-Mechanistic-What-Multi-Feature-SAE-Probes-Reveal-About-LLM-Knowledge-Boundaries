# FPE14 Frozen Dual-Output Confirmation Protocol

Protocol freeze date: 2026-07-17. Frozen before target selection and before any
FPE14 generation.

## Confirmatory Question

Can the fixed FPE13 rotation-invariant residual-energy readout replicate on new
questions, and do its global-hazard and within-question early-outcome effects
generalize across Qwen2.5-1.5B-Instruct and Gemma-2-2B?

## Frozen Target

- TriviaQA `rc.nocontext` validation split from the local official parquet.
- Exclude every normalized prompt in all existing local TriviaQA JSONL files.
- Deduplicate by both normalized prompt and dataset question ID before sampling.
- Select 400 unique questions with seed 20260718, balanced between 200
  `WikipediaEntity` answers and 200 other answer types.
- Eight sampled trajectories per question, temperature 0.7, top-p 0.9, top-k
  50, and at most 16 new tokens.
- Qwen generation seed 20260804; Gemma generation seed 20260805.
- No question may be replaced after either model output is observed.

## Frozen Models and Sources

1. `Qwen/Qwen2.5-1.5B-Instruct`, chat template, normalized layer 19. Source is
   the frozen FPE11 200-question 2Wiki collection.
2. `google/gemma-2-2b` base model, original shortest-answer prompt, layer 18.
   Source is the inherited pooled 2Wiki A+B collection. GemmaScope SAE latents
   are encoded only for the inherited representation control.

## Frozen Dual Output

Both outputs read T1 and use the same fixed eleven FPE13 geometric summaries:
RMS delta, mean absolute delta, maximum absolute delta, signed mean, standard
deviation, initial/current cosine, radial projection, relative delta norm, norm
ratio, clipped skewness, and clipped excess kurtosis. They are concatenated with
the ten inherited confidence covariates and fitted with the inherited ridge
logistic readout. No feature selection or target fitting is allowed.

- **Global hazard output:** trained on source next-token hazard among T1 risk
  rows; evaluated primarily by target global AUROC/AUPRC.
- **Early outcome output:** trained on source final error among all valid T1
  rows; evaluated primarily by macro within-question AUROC over questions with
  both final outcomes.

The outputs remain separate. No target-selected mixing coefficient or scalar
fusion is permitted.

## Frozen Comparisons

- Confidence only.
- Full dense residual plus confidence.
- Raw top-K with K=8.
- PCA-8, Gaussian RP-8, L1 raw-8, and 20 random raw-coordinate supports.
- SAE top-K for Gemma only.
- Deterministic-prefix observability upper bound.

## Confirmation Criteria

- At least 30 T1 hazard-discordant questions for a strong hazard within-question
  interpretation and at least 30 final-discordant questions for early outcome.
- Global hazard confirmation: invariant energy plus confidence exceeds
  confidence by at least 0.02 and the question-grouped 95% interval is not
  wholly negative in both models.
- Early-outcome confirmation: invariant energy plus confidence exceeds
  confidence by at least 0.03 within question and the grouped 95% interval is
  not wholly negative in both models.
- Compression retention for raw K=8: no more than 0.03 global AUROC below dense.
- SAE specificity is supported only if SAE top-K exceeds raw top-K, PCA-8, and
  the random-coordinate 95th percentile. Otherwise SAE remains an interpretable
  coordinate system rather than a uniquely predictive representation.
- Calibration and 10% FPR control use the unchanged FPE10 rules; wide intervals
  remain failures.

Point-estimate success without adequate discordant-question support is reported
as limited evidence. Any failure is retained and cannot trigger changes to the
features, layer, K, questions, source, penalty, or endpoint.

