# When, Where, and How Should Hallucination Risk Be Monitored?

Official code and frozen experimental artifacts for:

> **When, Where, and How Should Hallucination Risk Be Monitored? A
> Three-Property Framework for Internal Readout**

This repository studies what a deterministic monitor can legitimately infer
from a language model's current hidden state. It separates three nested claims:

1. **Risk observability:** outcome-related information is present in the current
   state.
2. **Conditional quantitative readoutability:** a prespecified readout can turn
   that information into a held-out score under a stated model, stage, and
   information regime.
3. **Support identifiability:** a selected representational support is stable,
   reproducible, and uniquely privileged.

Sparse autoencoder (SAE) features are one restricted readout family in this
framework. A successful SAE probe is not treated as evidence for a unique or
causal "hallucination neuron."

## Main result

Exact prefix collisions impose an information boundary: continuations that
share the same question and prefix have the same deterministic state, so a
state-only monitor cannot rank their still-unrealized outcomes. Once a candidate
answer is committed and appended to the same question and evidence, its
candidate-evidence compatibility becomes readable. This motivates the
**Commitment-Evidence Verification Router (CEVR)**, which selects among realized
candidates rather than predicting an unrealized random branch.

In the frozen, document-source-ID-isolated HotpotQA confirmation, the rank-32 dense CEVR
adapter improves first-sample accuracy from **33.75% to 69.38%**. This is a
bounded candidate-selection result, not a pre-generation safety guarantee. The
advantage over the strongest deduplicated-candidate logistic baseline remains
statistically uncertain, and the SAE router does not add an independent gain.
On the same candidate pool, a source-tuned 0.5B text cross-encoder reaches
40.63%, whereas the simpler dense unique-candidate CEVR reaches 66.88%. This
retrospective fairness audit is limited to the tested verifier scale and does
not establish superiority over all textual evidence verifiers.

## Repository scope

This public release contains only research code and artifacts needed to audit
the paper's claims:

```text
configs/       Model and SAE configuration templates
data/          Dataset access instructions and source-ID manifests
protocols/     Frozen confirmatory protocols
results/       Machine-readable frozen result records
scripts/       Data preparation, state analysis, controls, and CEVR evaluation
src/           Shared model loading, activation extraction, and probe utilities
tests/         CPU-only structural and import checks
```

The repository intentionally excludes manuscript-building code, figure and
table generation scripts, Word/LaTeX tooling, temporary audits, browser caches,
model checkpoints, SAE weights, raw third-party datasets, and exploratory
outputs that do not support a reported claim. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md).

## Installation

The reference environment uses Python 3.11 and CUDA 12.8. Create the environment
with Conda:

```bash
conda env create -f environment.yml
conda activate cevr-monitoring
```

PyTorch wheels are platform-specific. If Conda does not install the desired CUDA
build, install PyTorch first using the command recommended at
<https://pytorch.org/get-started/locally/>, then run:

```bash
pip install -r requirements.txt
```

Accept the model licence and authenticate with Hugging Face before downloading
Gemma or Qwen checkpoints. Model and SAE assets are never redistributed here.

## Data and cached representations

The experiments use public PopQA, TriviaQA, 2WikiMultiHopQA, HotpotQA, MuSiQue,
WebQuestions, and SQuAD records. Download these datasets from their official
distribution channels or through the Hugging Face `datasets` package, subject
to their original licences.

GitHub contains only:

- deterministic preparation code;
- compact source-ID manifests for split/isolation audits;
- frozen protocols and machine-readable aggregate results.

Hidden-state caches are reproducible research outputs but are too large for the
source repository. Their expected filenames and SHA-256 checksums are listed in
[`data/ARTIFACT_MANIFEST.csv`](data/ARTIFACT_MANIFEST.csv). Until a DOI-backed
archive is attached, regenerate them with the extraction scripts described in
[`data/README.md`](data/README.md).

## Reproduction map

| Claim | Entry point | Frozen artifact |
|---|---|---|
| Shared T0 state cannot identify a future branch | `scripts/poc_v43_t0_target_separation.py` | `results/t0_target_separation_*.json` |
| Readoutability is conditional | `scripts/poc_nn1_sparse_compression_controls.py` | `results/quantitative_readout_*.json` |
| Sparse supports need not be identifiable | `scripts/poc_nn12_information_regime_transport.py` | `results/support_transport_*.json` |
| Candidate-context compatibility appears after commitment | `scripts/poc_v40_extract_and_evaluate.py` and controls | `results/commitment_*.json` |
| Frozen CEVR improves over the first sample | `scripts/poc_v45_frozen_cevr_adapter_confirmation.py` | `results/frozen_cevr_confirmation.json` |
| Nonlinear adapter gain is not just parameter count | `scripts/poc_v46_nonlinear_adapter_attribution.py` | `results/nonlinear_adapter_attribution.json` |
| Prefix-stage monitoring can trade accuracy for token cost | `scripts/poc_nn36_stage_budget_intervention.py` | `results/stage_budget_intervention.json` |
| Equal-supervision text-only fairness audit | `scripts/poc_text_only_cevr_baseline.py`, `scripts/poc_frozen_text_cross_encoder.py`, and `scripts/poc_finetuned_text_cross_encoder.py` | `results/*text*cross_encoder.json` and `results/text_only_cevr_baseline.json` |

The exact command lines and required assets are documented in
[`REPRODUCE.md`](REPRODUCE.md). The internal development labels in original
filenames are retained only to preserve provenance; they are not manuscript
version numbers.

## Quick verification

The following checks do not download a model or dataset:

```bash
python -m compileall src scripts tests
python tests/test_release_integrity.py
python scripts/verify_frozen_results.py
```

## Reproducibility boundaries

- Training, selection, and evaluation partitions are grouped by question.
- Confirmatory protocols are stored in `protocols/` with freeze dates.
- Results are reported even when the tested method does not improve.
- The 7B experiment uses an NF4 checkpoint and a locally trained SAE; it is not
  presented as a second public-SAE replication.
- Dimensional compression is not reported as end-to-end speedup.
- CEVR requires a finite candidate set and supplied evidence.

## Citation

The manuscript is under review. Until a DOI is available, cite the repository
using [`CITATION.cff`](CITATION.cff). A versioned archival DOI will be added
before the final data-availability statement is frozen.

## Licence

Code in this repository is released under the MIT License. Dataset text, model
weights, SAE dictionaries, and third-party software retain their original
licences and are not relicensed by this repository.
