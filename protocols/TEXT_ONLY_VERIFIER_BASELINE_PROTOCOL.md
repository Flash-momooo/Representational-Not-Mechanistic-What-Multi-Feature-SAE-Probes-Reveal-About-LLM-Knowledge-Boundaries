# Frozen direct-text verifier baseline protocol

Frozen: 2026-08-22, before running the target evaluation implemented by
`scripts/poc_text_only_cevr_baseline.py`.

## Question

Does a supervised verifier that reads only the question, supplied evidence,
and realized candidate text recover the CEVR selection gain without access to
the monitored model's cached residual or SAE state?

## Data and candidate handling

- Source: the same clean V40f/2Wiki pool used by CEVR: 160 questions and eight
  sampled commitments per question before the label-blind phrase-leakage
  filter.
- Target: the frozen V45 HotpotQA pool: 160 questions and eight sampled
  commitments per question.
- Repeated samples selecting the same option ID are collapsed exactly as in
  CEVR. No gold-aware semantic clustering or human/LLM adjudication is used.
- Training is restricted to source questions containing both correct and
  incorrect realized options, matching the CEVR fitting population.
- Target labels are used only once for final evaluation and bootstrap
  intervals. No target refitting, threshold selection, vocabulary selection,
  or hyperparameter tuning is permitted.

## Text interface

Each candidate is represented only as:

`[QUESTION] question [EVIDENCE] evidence [CANDIDATE] candidate_text`

The verifier is a union of word 1--2 gram and character 3--5 gram TF-IDF
features followed by class-balanced L2 logistic regression. The candidate ID,
gold option, model activations, sampled likelihood, and output confidence are
excluded.

## Source-only model selection

Five-fold GroupKFold by question selects `C` from `{0.1, 1.0, 10.0}` using
mean held-out question-level selection accuracy. Ties select the smaller `C`.
All vectorizers and classifiers are refitted inside each fold.

## Endpoints

Primary: target question-level selection accuracy and paired 10,000-resample
interval versus the dense option-deduplicated linear CEVR.

Secondary: intervals versus restricted likelihood and first sampling,
candidate-population AUROC/AUPRC, and mean within-question AUROC.

## Claim boundary

This is a direct supervised text baseline, not a pretrained Transformer
cross-encoder or NLI verifier. A positive result would weaken any claim that
internal states are uniquely necessary. A negative result would establish an
increment over this declared text baseline, not over all possible textual
verifiers.
