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

The reference workstation used an NVIDIA GeForce RTX 5080 Laptop GPU. Gemma-2-2B
experiments fit on a 16 GB GPU in bfloat16. The Qwen2.5-7B experiment uses NF4
quantization. CPU-only scripts can recompute metrics from released caches.

