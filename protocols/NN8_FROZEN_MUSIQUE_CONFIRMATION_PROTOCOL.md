# NN8 Frozen MuSiQue Cross-Dataset Confirmation Protocol

Freeze date: 2026-07-26. Written before downloading or inspecting MuSiQue
question contents and before generating any NN8 model output.

## Purpose

Test whether the NN3 low-dimensional early-risk result transfers from
single-hop factual questions to an untouched 2--4 hop dataset. MuSiQue-Ans is
used because its questions are constructed by connected single-hop
composition, providing a structural shift beyond another TriviaQA subset.

## Frozen Data

- Source: official-format `musique_ans_v1.0_dev.jsonl` mirrored by
  `bdsaglam/musique`; record the immutable Hub revision and local SHA-256.
- License/source provenance is cross-checked against the official
  `StonyBrookNLP/musique` repository and TACL paper.
- Exclude every normalized prompt already present in project JSONL files.
- Deduplicate by normalized prompt and source question ID.
- Select 240 questions with seed 20260728, stratified as 80 each from 2-hop,
  3-hop, and 4-hop IDs. If a stratum has fewer than 80 eligible rows, retain
  all of that stratum and deterministically fill the deficit from the pooled
  eligible remainder before any generation.
- Use question-only shortest-answer prompting. Context paragraphs and question
  decompositions are not exposed to the model.
- No selected question may be replaced after observing model output.

## Frozen Models And Generation

1. Qwen2.5-1.5B-Instruct, chat template, normalized-depth layer 19, seed
   20260931.
2. Gemma-2-2B base, plain shortest-answer prompt, layer 18, seed 20260932;
   GemmaScope 16K latents are encoded as a secondary representation.

For each model and question, generate eight trajectories at temperature 0.7,
top-p 0.9, top-k 50, and at most 16 new tokens. Correctness is evaluated
against the official answer and aliases using the unchanged project scorer.

## Frozen Primary Analysis

- Use the unchanged NN3 five pair-ID-grouped folds, first-divergence T1--T3
  endpoint, K in {1, 2, 4, 8, 16, 32, 64}, rotating isolated selection fold,
  question-grouped NLL selection, and stage-specific refitting.
- Primary comparison is legacy adaptive supervised raw coordinates versus the
  legacy dense residual readout.
- Report confidence, PCA, random projection, SAE (Gemma), Brier, AUPRC,
  question-grouped bootstrap, selected K, and pre-event utility as secondary.
- Per-model compression retention uses the unchanged NN3 tolerances: AUROC
  delta 95% lower bound >= -0.02 and Brier delta 95% upper bound <= +0.01.
- Passing both models is the NN8 cross-dataset confirmation criterion. Failure
  is retained and cannot trigger question replacement, feature changes, or
  revised thresholds.

## Strong-Baseline Follow-Up

After the frozen NN8 primary analysis, apply the already frozen NN6A strong
baseline suite unchanged. It is confirmatory with respect to NN8 data because
its methods and grids were fixed before MuSiQue contents or outputs were seen,
but the NN8 primary pass/fail remains the inherited NN3 criterion above.
