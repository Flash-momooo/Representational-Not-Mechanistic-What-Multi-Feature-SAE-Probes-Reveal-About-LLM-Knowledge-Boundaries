# NN36 Stage-Wise Budgeted Intervention Protocol

## Status and purpose

NN36 is a retrospective, cache-only constructive experiment. It is designed to
test whether the temporal distinctions established by the three-property
framework can improve a real decision objective under a finite generation
budget. It is not a new prospective confirmation and it does not modify the
frozen CEVR result.

The experiment uses the previously frozen FPE14 TriviaQA trajectory caches:

- 400 questions and eight stochastic trajectories per question;
- Gemma-2-2B with GemmaScope SAE states;
- Qwen2.5-1.5B-Instruct with dense residual states;
- stages T0, T1, T2 and T3, final correctness, and generated-token counts.

## Decision problem

The intervention has two distinct parts.

1. T0 estimates question-level stochastic risk. It is used only to allocate a
   candidate budget from K in {1, 2, 4, 8}; it is never used to identify which
   unrealized branch will fail.
2. T1--T3 estimate the risk of an already realized prefix. A trajectory may be
   stopped and replaced when its score crosses a calibration-only threshold.

Completed candidates are selected by normalized-answer consensus. Ties are
broken by the monitor score, or by mean selected-token log probability for
no-monitor controls. If all candidates are stopped, the least risky stopped
trajectory is resumed; its remaining token cost is counted.

The T0 allocation score is the predicted Bernoulli variance p(1-p), not raw
error probability. Calibration-fold quartiles map increasing uncertainty to
K={1,2,4,8}. This avoids automatically spending the largest budget on
questions predicted to be almost certainly wrong, where repeated sampling has
little estimated marginal value.

## Isolation

Questions are assigned to five deterministic folds. For each outer test fold:

- three folds fit all readouts;
- one disjoint fold calibrates thresholds and T0 budget cut points;
- one fold is used only for policy evaluation.

No test label is used for fitting, threshold selection, budget assignment, or
candidate selection. All eight trajectories from one question remain in the
same fold.

## Readouts and controls

Stage-wise readouts are fixed before evaluation:

- random gate;
- confidence statistics;
- observed prefix hash plus confidence;
- dense residual change plus the same observed covariates;
- SAE change plus the same observed covariates, when SAE states are available.

The random gate uses deterministic independent scores and the same nominal
correct-trajectory false-abort rate. Fixed K={1,2,4,8} consensus systems form
the accuracy--cost baseline. T0 adaptive allocation is compared with fixed K=4
and a random allocation with the same candidate-budget support.

## Frozen operating points and metrics

The threshold grid is a correct-trajectory false-abort rate of 5%, 10% and
20%. The 10% point is primary. No operating point will be selected using test
performance.

Primary metrics are:

- answer accuracy at the realized mean generated-token cost;
- mean generated tokens per question;
- paired accuracy and token-cost differences from fixed K=4 and matched random
  gating;
- strictly pre-event recall among erroneous trajectories;
- mean token lead for strictly pre-event alerts.

Secondary outputs include retained-candidate count, attainable any-correct
oracle under the allocated K, and the full accuracy--cost table. Confidence
intervals use question-grouped bootstrap resampling.

## Interpretation boundary

A positive result would show that stage-conditioned risk readout can be turned
into a budgeted intervention under the tested decoding policy. It would not
show that T0 predicts branch identity, that SAE is universally optimal, or that
the monitor supplies a safety guarantee. A negative result would distinguish
readability from operational utility and would remain informative for the
framework.
