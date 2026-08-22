# Source-tuned text cross-encoder fairness protocol

Frozen: 2026-08-22, after the frozen-encoder audit and before target execution.

## Objective

Provide an equal-supervision textual verifier that can adapt the pretrained
encoder, rather than only fitting a linear head to frozen text features.

## Fixed model and input

- Model: `Qwen/Qwen2.5-0.5B-Instruct` with a two-class sequence-classification
  head; all parameters are updated on source data.
- Text: the same question, supplied evidence, and normalized candidate string
  used in the preceding text audits. Option letters, gold identity, generator
  likelihood, and monitored Gemma states are excluded.
- Maximum length: 1,024 tokens with left truncation; batch size 2 and gradient
  accumulation 8; AdamW, learning rate `2e-5`, weight decay `0.01`.

## Numerical amendment before a valid target result

The first execution stored model parameters and AdamW states in bfloat16. The
source-only epoch selection completed, but the final target scores contained
NaN values, so no target metric was valid or recorded. Before obtaining a
target result, the implementation was amended to keep parameters, gradients,
and optimizer states in float32 while using bfloat16 automatic mixed precision
for Transformer forward computation. Data, split, seed, learning rate, epoch
selection, candidate handling, and endpoints are unchanged.

## Source-only selection

The same 357 option-deduplicated candidates from 114 mixed 2Wiki questions are
split once by question with `GroupShuffleSplit(test_size=0.2,
random_state=20260822)`. Training runs for three epochs. The earliest epoch with
the highest held-out question-level selection accuracy is selected. A fresh
model is then trained on all source candidates for exactly that many epochs.
No HotpotQA label or statistic participates in this choice.

## Target evaluation

The final model scores the same 472 realized candidates from 160 HotpotQA
questions. Primary and secondary paired comparisons match the frozen-encoder
protocol and use 10,000 question bootstrap resamples.

## Interpretation

This is an exploratory, source-tuned 0.5B cross-encoder baseline. It is stronger
than TF-IDF and a frozen encoder, but it does not exhaust larger NLI models,
instruction tuning, or target-domain supervision. The target pool was already
used in earlier CEVR analyses, so the audit is retrospective.
