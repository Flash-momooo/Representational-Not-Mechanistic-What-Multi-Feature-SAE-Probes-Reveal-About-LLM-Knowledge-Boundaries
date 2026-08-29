# Framework Extension: Frozen Records and Audit Status

This release records the experiments added after the initial 160-question
candidate-routing result. It distinguishes a prospective scale confirmation
from retrospective diagnostic audits. These labels are part of the scientific
claim, not a presentation detail.

## V48: candidate-conditioned scale confirmation

- **Target:** 500 source-ID-isolated HotpotQA questions, with four supplied,
  evidence-grounded candidates per question (2,000 candidate states).
- **Source fit:** a fixed 160-question 2Wiki real-condition cache; no target
  refit, layer selection, feature selection, or threshold fitting.
- **Backbone/state:** Gemma-2-2B layer-18 residual state after each candidate
  has entered the same question--evidence context.
- **Endpoints:** question-grouped within-question AUROC and four-candidate
  selection accuracy, both with a 10,000-question bootstrap.
- **Boundary:** this confirms candidate-conditioned readout. It is not a
  multi-sample generation trial, a pre-generation branch prediction result, or
  a causal-localization result.

## V49, V53, and V54: diagnostic audits of the V48 target

- **V49** compares task-aligned verifiers without refitting V48 state scores.
  The external Qwen-7B verifier uses a distinct model, prompt, and resource
  regime and is reported separately from same-backbone comparisons.
- **V53** tests whether non-identical SAE feature supports can nevertheless
  span aligned, useful candidate-relative subspaces. Support selection uses
  source questions only; the V48 target is already examined, so this is
  diagnostic rather than prospective.
- **V54** compares Dense, SAE, PCA, candidate likelihood, post-candidate token
  entropy, and a residual-norm control using the same 2,000 Gemma forwards.
  It cannot establish a universal interface ranking.

## V51 and V52: geometric audits at distinct scales

- **V51** uses a fixed Qwen2.5-7B Top-K SAE and 600 WebQuestions trajectories
  to separate coordinate overlap from decoder-subspace alignment. It is not a
  cross-model result or a causal-circuit test.
- **V52** evaluates 2,000 questions and 16,000 trajectories across Qwen2.5-7B,
  Gemma-2-2B, and Qwen2.5-1.5B. It reports question-centered controls because
  uncentered geometry can reflect question difficulty rather than branch risk.

## V55--V56: genuine stochastic-output audit

This separate audit contains 80 HotpotQA questions with eight independently
sampled Gemma-2-2B-IT answers each. Semantic-entropy-style clustering and a
SelfCheck-style disagreement score operate on genuine samples, never on the
four fixed V48 answer options. The 0.5B and 7B local judges are reproducible
approximations, not verbatim reproductions of every semantic-entropy or
SelfCheckGPT implementation.

## Artifact handling

Aggregate outputs are committed under `results/`. Generated trajectories,
hidden-state caches, and the locally trained Qwen SAE are not redistributed;
their expected checksums appear in `data/ARTIFACT_MANIFEST.csv`. Public model
and dataset licences continue to govern regeneration.
