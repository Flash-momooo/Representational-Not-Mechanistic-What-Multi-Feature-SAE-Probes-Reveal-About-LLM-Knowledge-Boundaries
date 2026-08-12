# FPE20 Frozen 7B+ Policy-Observability Replication

Freeze date: 2026-07-26. This protocol was written before downloading the
checkpoint or generating any FPE20 trajectory. Results may not change the
model, question rows, decoding policies, sample count, label rule, bootstrap,
or success criterion below.

## Purpose

Test whether the FPE15/FPE18 observability result extends beyond 1.5B--2B
models. This is a model-scale external-validity replication, not an SAE
superiority experiment. No SAE is required because the deterministic-prefix
bound applies to every state-only readout.

## Frozen Design

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Loading: bitsandbytes NF4 four-bit weights, double quantization, bfloat16
  computation. The exact configuration is fixed for all conditions.
- Questions: rows 1--200 of
  `data/fpe14_trivia_confirmatory_questions.jsonl`, identical to FPE15/FPE18.
- Prompt: model chat template and the frozen shortest-answer instruction.
- Trajectories: 16 per question and condition.
- Temperatures: 0.2, 0.7, and 1.0.
- Fixed top-p 0.9, top-k 50, maximum 16 new tokens.
- Common seed: 20260901 plus question index in every temperature condition.
- Correctness and T0--T3 prefix definitions: unchanged from FPE15/FPE18.
- Uncertainty: 10,000 question-level bootstrap samples.

The quantized checkpoint is used because the available GPU has 16GB memory.
The result is evidence for the 7B model configuration actually evaluated and
must not be described as a full-precision invariance result.

## Frozen Criterion

The replication passes only if:

1. conditional entropy is nonincreasing, prefix information is nondecreasing,
   tied opposite-outcome fraction is nonincreasing, and the deterministic
   prefix AUROC ceiling is nondecreasing in every temperature condition; and
2. between temperature 0.2 and 1.0, either the error-rate difference or the
   within-question outcome-entropy difference has a 95% question-bootstrap
   interval excluding zero.

No directional effect or minimum magnitude is required. A null policy effect
or a failed structural audit will be retained and reported. No detector,
position, layer, threshold, or calibration model may be selected from FPE20.

## Outputs

- `data/fpe20_qwen7b_t02_trajectory.jsonl`
- `data/fpe20_qwen7b_t07_trajectory.jsonl`
- `data/fpe20_qwen7b_t10_trajectory.jsonl`
- `outputs/poc_fpe20_qwen7b_observability_results.json`
