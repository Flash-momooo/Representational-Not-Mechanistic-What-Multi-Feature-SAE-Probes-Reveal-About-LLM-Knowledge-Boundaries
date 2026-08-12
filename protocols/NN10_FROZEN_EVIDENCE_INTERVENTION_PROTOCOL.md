# NN10 Frozen MuSiQue Evidence Intervention Protocol

Freeze date: 2026-07-26. Written after retaining the failed question-only NN8
confirmation and before constructing evidence prompts or generating NN10
outputs.

## First-Principles Question

NN8 confounds monitor transfer with task solvability: both small models almost
always fail without multihop evidence. NN10 intervenes on information
availability while holding question identity, gold answers, models, sampling,
and analysis fixed. It asks whether sparse risk retention becomes identifiable
when the model is given the official evidence required to solve the same task.

## Frozen Intervention

- Reuse all 240 NN8 questions without replacement.
- Join each source ID to the immutable MuSiQue-Ans dev file and include every
  paragraph marked `is_supporting`, in official source order, with its title.
- Do not expose question decomposition, intermediate answers, distractor
  paragraphs, or answer labels.
- Prompt: `Answer using only the evidence below. Return only the shortest
  answer with no explanation`, followed by evidence, question, and `Answer:`.
- Assert every prompt fits each model context window with 16 generation tokens;
  do not truncate examples after seeing outputs.

## Frozen Models And Sampling

- Qwen2.5-1.5B-Instruct, chat template, layer 19, seed 20260941.
- Gemma-2-2B base plus GemmaScope 16K, layer 18, seed 20260942.
- Eight trajectories per question, temperature 0.7, top-p 0.9, top-k 50,
  maximum 16 new tokens.

## Frozen Analysis And Adequacy

- Apply the unchanged NN3 primary algorithm and pass tolerances.
- Apply the already frozen NN6A strong-baseline suite unchanged.
- Before interpreting operational FPR, require at least 96 correct trajectories
  and at least 20 discordant questions per model. Below either threshold, report
  the model as support-inadequate even if AUROC is numerically available.
- Evidence-conditioned cross-dataset retention passes only if both models meet
  support adequacy and both pass the inherited sparse-versus-dense AUROC/Brier
  tolerances.
- Compare NN10 to NN8 descriptively as an information-availability intervention;
  no paired causal claim about hidden states is made because generated outputs
  differ across conditions.
