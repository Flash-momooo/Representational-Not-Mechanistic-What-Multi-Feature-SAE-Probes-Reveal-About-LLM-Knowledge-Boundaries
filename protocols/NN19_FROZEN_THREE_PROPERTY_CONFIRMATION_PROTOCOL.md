# NN19 Frozen Three-Property Monitoring Confirmation Protocol

Freeze date: 2026-07-29. This protocol is written after development on the
already used Qwen-7B/TriviaQA condition and before downloading or inspecting
the NN19 source records, selecting NN19 questions, generating an NN19 model
continuation, extracting an NN19 hidden state, or seeing an NN19 label.

## Development-Derived, Frozen Hypothesis

The development condition showed three facts: no trajectory-specific alert is
possible at the shared T0 prefix; adaptive sparse coordinates were useful at
the first observable prefix; and their adjacent-stage support overlap was low
(K=64 mean Jaccard about 0.014). The resulting *stage-structured PARR*
(SS-PARR) schedule is frozen as:

1. no alert at T0 (observability gate);
2. adaptive sparse raw-residual readout at T1 (local compressibility);
3. dense raw-residual readout at T2 and T3 (non-identifiable sparse support
   across stages).

This is a single fixed schedule, not a per-test-fold search over schedules.
It tests whether the three properties supply a useful structural inductive
bias beyond fixed readouts and free NLL routing.

## Frozen Data And Model

- Source: public `web_questions` test split, a factual open-domain QA source
  not used by any earlier project experiment.
- Exclude every normalized question, source ID, and `(question, answer)` pair
  in pre-existing project JSONL files. Deterministically shuffle the eligible
  source records with seed 20260730 and retain the first 600. No replacement
  is permitted after generation begins.
- Eight trajectories/question, temperature 0.7, top-p 0.9, top-k 50, maximum
  16 new tokens, seed 20260734; unchanged shortest-answer chat prompt and
  scorer.
- Model: local `models/Qwen2.5-7B-Instruct`, NF4 quantized because the full
  checkpoint exceeds available GPU memory. Layer: normalized depth 18/26 as
  resolved from the actual model layer count. Quantization is reported as a
  limitation and NN19 makes no full-precision scale claim.

## Frozen Analysis

- Use five question-ID-grouped outer folds with seed 20260726. For test fold
  `f`, fold `(f + 1) mod 5` selects K from {1,2,4,8,16,32,64}; the other three
  non-test folds fit support and readouts. After K selection, refit on all
  non-test rows. No previous result file is read.
- Compare SS-PARR against fixed adaptive sparse, fixed dense, confidence, and
  free stagewise NLL PARR. The latter may select among confidence, sparse, and
  dense separately at each stage using only the selection fold.
- Trajectory risk is the maximum available T1--T3 probability. A nominal 10%
  threshold is fitted on verified-success trajectories in rotating calibration
  fold `(f + 2) mod 5`, never the test fold. First crossing is the alert.

## Frozen Criteria

NN19 is calibration-adequate only if each threshold-calibration fold contains
at least 96 verified-success trajectories. Otherwise its thresholded endpoint
is reported as inadequate, without changing data, model, or threshold.

Primary endpoint: question-grouped bootstrap 95% lower interval above zero for
SS-PARR minus fixed adaptive sparse strictly pre-event recall, while the
SS-PARR-minus-sparse point actual-FPR difference is no more than +0.03. Both
must hold, and calibration adequacy must hold. AUROC, AUPRC, Brier, NLL,
lead-time, route frequencies, selected K, and comparisons to dense and free
PARR are secondary. A failure remains part of the record.

## Claim Scope

Passing would support a narrow system claim: an observability gate plus a
pre-specified sparse-to-dense transition can improve timely risk monitoring in
one new QA source under a quantized 7B model. It would not prove that sparse
coordinates are causal hallucination mechanisms, that the schedule is
universal, or that the monitor is a certified safety controller.
