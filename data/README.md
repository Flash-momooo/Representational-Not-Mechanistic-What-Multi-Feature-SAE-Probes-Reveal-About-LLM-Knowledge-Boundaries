# Data and Artifact Access

## Public source datasets

The study reuses PopQA, TriviaQA, 2WikiMultiHopQA, HotpotQA, MuSiQue,
WebQuestions, and SQuAD. Obtain each dataset from its official distribution or
through the Hugging Face `datasets` package. Users are responsible for complying
with each dataset's licence and terms.

No raw third-party dataset is committed to this repository.

## Public files in this repository

`manifests/` contains compact source identifiers and split metadata used to
audit prospective source isolation. These manifests intentionally omit evidence
paragraphs, prompts, and answer text.

`../results/` contains aggregate machine-readable outputs underlying reported
tables and claims. It does not contain model weights or private user data.

## External research artifacts

Exact numerical reproduction uses hidden-state `.npz` caches and model-generated
trajectory files. The expected files, byte sizes, roles, and SHA-256 checksums
are listed in `ARTIFACT_MANIFEST.csv`.

These artifacts are excluded from Git because they are large and may contain
derived text from third-party datasets. Before journal submission, they should
be deposited in a versioned research-data repository that supports a DOI and,
if necessary, reviewer-only access. The DOI is currently unresolved and is not
fabricated here.

## Regeneration

Preparation and extraction scripts are included under `../scripts/`. Default
local cache directories are ignored by Git. See `../REPRODUCE.md` for commands.

