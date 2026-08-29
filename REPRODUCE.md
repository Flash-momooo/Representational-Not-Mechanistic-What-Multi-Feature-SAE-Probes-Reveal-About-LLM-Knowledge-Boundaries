# Reproduction Guide

Run commands from the repository root. Paths below assume the artifact names in
`data/ARTIFACT_MANIFEST.csv`.

## Level 0: audit released results

No GPU, model, or dataset is required:

```bash
python scripts/verify_frozen_results.py
python tests/test_release_integrity.py
```

This verifies the headline numbers, confidence interval ordering, source-ID
isolation summary, expected result schemas, and the absence of forbidden
publication-tool files.

## Level 1: rerun cache-only analyses

Place the required `.npz` caches under `artifacts/cache/` and the associated
generation records under `artifacts/generation/`. Verify their SHA-256 values
against `data/ARTIFACT_MANIFEST.csv`.

### T0 question risk versus future-branch identity

```bash
python scripts/poc_v43_t0_target_separation.py \
  --cache artifacts/cache/fpe14_gemma_trivia_confirmatory_trajectory_states.npz \
  --output reproduced/t0_gemma.json
```

Repeat with the Qwen-1.5B TriviaQA cache for the cross-model audit.

### Conditional quantitative readoutability

```bash
python scripts/poc_nn1_sparse_compression_controls.py \
  --data artifacts/trajectories/fpe14_gemma_trivia_confirmatory_trajectory.jsonl \
  --cache artifacts/cache/fpe14_gemma_trivia_confirmatory_trajectory_states.npz \
  --tokenizer google/gemma-2-2b \
  --output reproduced/readout_gemma_trivia.json \
  --cpu
```

### Support transport

```bash
python scripts/poc_nn12_information_regime_transport.py \
  --question-only-data artifacts/trajectories/nn8_gemma_trajectory.jsonl \
  --question-only-cache artifacts/cache/nn8_gemma_trajectory_states.npz \
  --evidence-data artifacts/trajectories/nn10_gemma_evidence_trajectory.jsonl \
  --evidence-cache artifacts/cache/nn10_gemma_evidence_trajectory_states.npz \
  --tokenizer google/gemma-2-2b \
  --output reproduced/support_transport_gemma.json \
  --cpu
```

### Frozen CEVR confirmation

```bash
python scripts/poc_v45_frozen_cevr_adapter_confirmation.py \
  --train-cache artifacts/cache/v40f_no_rationale_states.npz \
  --train-generation artifacts/generation/poc_v40b_commitment_generation_results.json \
  --target-cache artifacts/cache/v45_hotpot_no_rationale_states.npz \
  --target-generation artifacts/generation/poc_v45_hotpot_no_rationale_generation_results.json \
  --target-items artifacts/manifests/v45_hotpot_adapter_confirmation_candidates.jsonl \
  --output reproduced/frozen_cevr_confirmation.json
```

The frozen method is defined in
`protocols/V45_FROZEN_CEVR_ADAPTER_CONFIRMATION_PROTOCOL.md`.

### Stage-wise budgeted intervention

```bash
python scripts/poc_nn36_stage_budget_intervention.py \
  --output results/stage_budget_intervention.json
```

This cache-only analysis uses question-grouped outer folds and disjoint
training, calibration, and evaluation questions. Its frozen scope and
interpretation are recorded in
`protocols/NN36_STAGE_BUDGET_INTERVENTION_PROTOCOL.md`.

### Equal-supervision text verifier audit

The three text-only baselines use the same source labels and realized target
candidate pool as the CEVR comparison:

```bash
python scripts/poc_text_only_cevr_baseline.py
python scripts/poc_frozen_text_cross_encoder.py
python scripts/poc_finetuned_text_cross_encoder.py
```

The first command is CPU-compatible. The latter two require the locally cached
`Qwen/Qwen2.5-0.5B-Instruct` checkpoint; end-to-end fine-tuning additionally
requires a CUDA device. These are retrospective fairness audits, as stated in
their corresponding protocol files, rather than new prospective confirmations.

### Framework-extension artifacts (V48--V55)

The current workshop claims are backed by the frozen records below. They use
large caches and generated third-party-derived trajectories that are not
committed to Git; verify their checksums in `data/ARTIFACT_MANIFEST.csv` before
rerunning an analysis.

```bash
# 500-question candidate-conditioned confirmation
python scripts/poc_v48_scale_candidate_readout.py \
  --source-cache artifacts/cache/equal_compute_source.npz \
  --cache artifacts/cache/v48_hotpot_scale_candidate_readout.npz \
  --items artifacts/data/v47_hotpot_scale_confirmation_candidates.jsonl \
  --output reproduced/poc_v48_hotpot_scale_candidate_readout.json

# Candidate-conditioned SAE geometry and same-state interface controls
python scripts/poc_v53_candidate_conditioned_subspace_audit.py
python scripts/poc_v54_unified_interface_comparison.py

# Genuine eight-sample semantic-uncertainty / consistency audit
python scripts/poc_v55_multisample_semantic_baselines.py \
  --output reproduced/poc_v55_multisample_semantic_baselines.json
```

`poc_v49_*` is a post-hoc audit of the already inspected V48 target, and V53--V55
are diagnostic audits with the status stated in
`protocols/V48_V55_FRAMEWORK_EXTENSION_PROTOCOL.md`. V52 is a separate
larger-scale geometric validation. These distinctions are deliberate: the
repository does not relabel a diagnostic as a prospective confirmation.

## Level 2: regenerate model states

1. Download the public datasets using the official source or the `datasets`
   package.
2. Accept and download the requested model checkpoint under its original
   licence.
3. Download GemmaScope SAEs through `sae_lens`, or train the explicitly labeled
   Qwen SAE experiment using `scripts/poc_nn32_train_qwen7b_true_sae.py`.
4. Run the relevant preparation script.
5. Run extraction with `--force-extract` where supported.
6. Run the cache-only analysis command above.

Model generation is stochastic. Exact trajectory text may vary across hardware
and library versions. Frozen released caches are therefore the primary artifact
for exact numerical reproduction; regeneration is used to test procedural
reproducibility.

## Compute

Experiments were run in a single-GPU environment with an NVIDIA GeForce RTX 5080
mobile GPU (16 GB VRAM). Gemma-2-2B was evaluated in bfloat16, whereas the
Qwen2.5-7B experiment used NF4 quantization. CPU-only scripts can recompute
metrics from the released caches.
