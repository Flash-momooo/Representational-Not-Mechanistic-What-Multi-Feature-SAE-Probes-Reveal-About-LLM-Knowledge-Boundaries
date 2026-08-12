# Public Release Scope

This repository is a publication release, not a mirror of the full research
workspace. Files are selected by their role in reproducing or auditing a paper
claim.

## Included

- reusable model, SAE, hidden-state, and probe code;
- scripts used to prepare reported datasets and candidate sets;
- analysis scripts for observability, quantitative readoutability, support
  identifiability, matched controls, and CEVR;
- frozen protocols and final machine-readable result records;
- source-ID manifests and checksums required to audit split isolation;
- environment and smoke-test files.

## Excluded

- figure, heatmap, table, graphical-abstract, and manuscript generation code;
- Word, LaTeX, PDF, LibreOffice, Inkscape, and browser automation utilities;
- temporary inspection and one-off debugging scripts;
- historical manuscript versions and internal progress notes;
- model checkpoints, tokenizer files, SAE weights, and locally trained SAE
  checkpoints;
- Hugging Face caches, browser profiles, logs, PID files, and intermediate
  tensor dumps;
- raw or reconstructed third-party dataset text where redistribution rights are
  not established;
- exploratory experiments not used to support a manuscript claim.

## Why derived records are separated from source data

The analysis depends on public datasets and model-generated trajectories. The
source datasets remain governed by their original licences. The public code
rebuilds derived candidate sets deterministically from official sources, while
compact source-ID manifests document split isolation without republishing the
underlying text. Large hidden-state caches are listed by checksum and should be
released through a versioned research-data archive rather than committed to Git.

## Provenance

Original experimental filenames such as `poc_v45_*` are retained in `scripts/`
because the frozen protocols and result records refer to them. Public-facing
documentation groups the same files by scientific claim rather than by the
chronology of development.

