# V45 Frozen CEVR-Adapter Confirmation Protocol

Date frozen: 2026-08-08

## Objective

Prospectively test whether the V44 development process yields a post-commitment
candidate router that improves over the original linear CEVR on new source
records. V45 is confirmatory for the selected adapter, not for every V44
ablation.

## Frozen Method

The method is selected only by V40f / 2Wiki grouped out-of-fold question
accuracy. The frozen configuration is:

- one semantic entry per distinct sampled answer option;
- Gemma-2-2B base layer-18 post-commitment raw residual;
- absolute candidate state, standardized with V40f statistics;
- rank-32 nonlinear risk adapter;
- question-listwise correctness objective;
- three-seed score ensemble (20260808--20260810);
- 100 AdamW epochs, learning rate 1e-3, weight decay 1e-3, dropout 0.10;
- choose the sampled unique candidate with maximum predicted utility, breaking
  ties by the earliest sample.

No V45 label, feature statistic, or result may alter this configuration.

## New Target Set

- Source: HotpotQA distractor validation.
- Exactly 160 new questions.
- Every primary and distractor source ID must be absent from every pre-existing
  `data/*.jsonl` artifact when V45 is constructed.
- Source IDs may not be reused within V45.
- Each item contains one gold answer and three real evidence-supported
  distractors, shuffled before generation.
- Eight no-rationale restricted commitments are sampled with decoding
  temperature 0.7. The router may select only among options actually sampled.

## Comparisons

- Primary: frozen V45 adapter versus original sample-weighted linear CEVR.
- Strong secondary: frozen V45 adapter versus restricted candidate likelihood.
- Additional descriptive baselines: first sample, reachable oracle, and the
  unique-candidate centered logistic readout.

All comparisons use question-level paired 10,000-bootstrap intervals.

## Decision

The primary criterion passes if the lower bound of the paired 95% interval for
adapter accuracy minus original linear CEVR accuracy is greater than zero.
The strong secondary criterion is reported independently and passes only if
its lower bound is greater than zero.

The result supports source-level prospective replication of candidate routing
after commitment. Because V45 remains within HotpotQA and uses the same
Gemma-2-2B family, it cannot establish cross-task, cross-model, SAE-specific,
causal, calibrated-refusal, or pre-generation prediction claims.
