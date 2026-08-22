# Frozen pretrained text cross-encoder protocol

Frozen: 2026-08-22, after the direct TF-IDF baseline and before running
`scripts/poc_frozen_text_cross_encoder.py`.

## Purpose

Test whether a generic pretrained text encoder with the same source labels and
the same realized candidate pool can reproduce the candidate-selection gain of
the monitored-model residual interface.

## Fixed interface

- Encoder: locally cached `Qwen/Qwen2.5-0.5B-Instruct`; all Transformer weights
  remain frozen.
- Input: question, supplied evidence, and one realized candidate, jointly
  encoded in one sequence ending in a fixed verification query.
- Feature: the final non-padding token at the last Transformer layer.
- Maximum length: 1,536 tokens, left truncation, right padding.
- Classifier: class-balanced L2 logistic regression.
- Source-only selection: five-fold GroupKFold by question chooses
  `C in {0.1, 1.0, 10.0}` by held-out question-level candidate-selection
  accuracy, with smaller `C` breaking ties.

## Data fairness

The source and target pools, label-blind leakage filter, option-identity
deduplication, mixed-question fitting restriction, and evaluation endpoints are
identical to `TEXT_ONLY_VERIFIER_BASELINE_PROTOCOL_2026-08-22.md`. Candidate
likelihood, monitored Gemma residuals, Gemma SAE features, option letters, and
gold identity are excluded from the input.

## Endpoints

Primary: target selection accuracy and paired 10,000-resample interval versus
dense option-deduplicated linear CEVR. Secondary comparisons are against
restricted likelihood, first sampling, the rank-32 listwise CEVR, and the
direct TF-IDF verifier.

## Claim boundary

This is a frozen-encoder cross-encoded text baseline, not an end-to-end
fine-tuned NLI model. The target set was already part of earlier CEVR analyses,
so this is a retrospective fairness audit, not a prospective confirmation.
