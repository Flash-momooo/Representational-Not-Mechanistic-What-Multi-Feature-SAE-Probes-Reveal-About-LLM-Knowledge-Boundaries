"""Train an isolated small Top-K SAE on Qwen2.5-7B prompt residuals.

The frozen protocol is paper/NN32_FROZEN_QWEN7B_TRUE_SAE_PROTOCOL.md.
No WebQuestions trajectory, label, or answer is read in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.poc_fpe10_collect_generic import format_prompt, load_model, model_layers, read_jsonl
from src.extract import ResidualHook


LAYER = 18
N_PROMPTS = 240
N_TRAIN_PROMPTS = 192
EXPANSION = 2
TOP_K = 32
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
SEED = 20260730
MODEL = "models/Qwen2.5-7B-Instruct"
INPUT = ROOT / "data" / "fpe12_trivia_holdout_questions.jsonl"
CACHE = ROOT / "outputs" / "cache" / "nn32_qwen7b_sae_train_prompt_residuals.npz"
WEIGHTS = ROOT / "outputs" / "nn32_qwen7b_l18_topk_sae.pt"
RESULT = ROOT / "outputs" / "poc_nn32_qwen7b_true_sae_training.json"


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, top_k: int, mean: torch.Tensor):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Parameter(torch.empty(d_sae, d_model))
        self.register_buffer("mean", mean.float().clone())
        self.top_k = int(top_k)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder, std=0.02)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.div_(self.decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        preactivation = self.encoder(values.float() - self.mean)
        top_values, top_indices = torch.topk(preactivation, self.top_k, dim=-1)
        top_values = torch.relu(top_values)
        latent = torch.zeros_like(preactivation)
        latent.scatter_(1, top_indices, top_values)
        return latent

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(values)
        return latent @ self.decoder + self.mean, latent


def collect_prompt_residuals(force: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    if CACHE.exists() and not force:
        archive = np.load(CACHE)
        return archive["train"].astype(np.float32), archive["validation"].astype(np.float32), json.loads(str(archive["metadata"].item()))

    rows = read_jsonl(INPUT)[:N_PROMPTS]
    if len(rows) != N_PROMPTS:
        raise ValueError(f"Expected {N_PROMPTS} prompts, found {len(rows)}")
    model, tokenizer = load_model(MODEL, load_in_4bit=True)
    hook = ResidualHook().attach(model_layers(model)[LAYER])
    train_blocks, validation_blocks = [], []
    prompt_lengths = []
    try:
        for index, row in enumerate(tqdm(rows, desc="nn32-collect-prompt-residuals")):
            prompt = format_prompt(tokenizer, row["prompt"], chat_template=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            values = hook.value[0].float().cpu().numpy()
            prompt_lengths.append(int(values.shape[0]))
            (train_blocks if index < N_TRAIN_PROMPTS else validation_blocks).append(values)
    finally:
        hook.detach()
        del model
        torch.cuda.empty_cache()

    train = np.concatenate(train_blocks, axis=0).astype(np.float16)
    validation = np.concatenate(validation_blocks, axis=0).astype(np.float16)
    metadata = {
        "source": str(INPUT.relative_to(ROOT)),
        "model": MODEL,
        "load_in_4bit": True,
        "layer": LAYER,
        "n_prompts": N_PROMPTS,
        "n_train_prompts": N_TRAIN_PROMPTS,
        "n_validation_prompts": N_PROMPTS - N_TRAIN_PROMPTS,
        "n_train_tokens": int(train.shape[0]),
        "n_validation_tokens": int(validation.shape[0]),
        "hidden_size": int(train.shape[1]),
        "prompt_length_min": int(min(prompt_lengths)),
        "prompt_length_median": float(np.median(prompt_lengths)),
        "prompt_length_max": int(max(prompt_lengths)),
        "labels_or_answers_read": False,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, train=train, validation=validation, metadata=json.dumps(metadata))
    return train.astype(np.float32), validation.astype(np.float32), metadata


@torch.no_grad()
def evaluate(model: TopKSAE, values: torch.Tensor, batch_size: int = 512) -> dict:
    sq_error = total_variance = cosine_sum = l0_sum = 0.0
    n = 0
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        decoded, latent = model(batch)
        residual = batch - decoded
        sq_error += float(residual.square().sum().item())
        total_variance += float((batch - model.mean).square().sum().item())
        cosine_sum += float(torch.nn.functional.cosine_similarity(batch, decoded, dim=-1).sum().item())
        l0_sum += float((latent > 0).sum(dim=-1).sum().item())
        n += int(batch.shape[0])
    mse = sq_error / max(n * values.shape[1], 1)
    return {
        "mse": float(mse),
        "explained_variance": float(1.0 - sq_error / max(total_variance, 1e-12)),
        "mean_cosine": float(cosine_sum / max(n, 1)),
        "mean_l0": float(l0_sum / max(n, 1)),
    }


def train_sae(train_values: np.ndarray, validation_values: np.ndarray) -> dict:
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    train = torch.as_tensor(train_values, dtype=torch.float32, device=device)
    validation = torch.as_tensor(validation_values, dtype=torch.float32, device=device)
    mean = train.mean(dim=0)
    d_model = int(train.shape[1])
    sae = TopKSAE(d_model, d_model * EXPANSION, TOP_K, mean).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device=device).manual_seed(SEED)
    history = []
    for epoch in range(EPOCHS):
        permutation = torch.randperm(len(train), generator=generator, device=device)
        loss_sum, seen = 0.0, 0
        sae.train()
        for start in range(0, len(train), BATCH_SIZE):
            batch = train[permutation[start:start + BATCH_SIZE]]
            decoded, _ = sae(batch)
            loss = torch.mean((decoded - batch).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            loss_sum += float(loss.detach().item()) * len(batch)
            seen += len(batch)
        sae.eval()
        validation_audit = evaluate(sae, validation)
        history.append({"epoch": epoch + 1, "train_mse": loss_sum / max(seen, 1), **validation_audit})
        print(json.dumps(history[-1]))
    final_train = evaluate(sae, train)
    final_validation = evaluate(sae, validation)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"d_model": d_model, "d_sae": d_model * EXPANSION, "top_k": TOP_K, "layer": LAYER, "architecture": "TopKSAE"},
    }, WEIGHTS)
    return {"train": final_train, "validation": final_validation, "history": history, "d_model": d_model, "d_sae": d_model * EXPANSION}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-collect", action="store_true")
    args = parser.parse_args()
    train, validation, collection = collect_prompt_residuals(args.force_collect)
    trained = train_sae(train, validation)
    payload = {
        "experiment": "NN32 isolated true SAE training on Qwen2.5-7B-Instruct",
        "protocol": "paper/NN32_FROZEN_QWEN7B_TRUE_SAE_PROTOCOL.md",
        "collection": collection,
        "architecture": {"type": "TopKSAE", "expansion": EXPANSION, "top_k": TOP_K, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "seed": SEED},
        "audit": trained,
        "weights": str(WEIGHTS.relative_to(ROOT)),
        "passed_minimum_reconstruction_audit": bool(trained["validation"]["explained_variance"] > 0.0 and trained["validation"]["mean_l0"] > 0.0),
        "interpretation": "A small independently trained SAE, not an official QwenScope-quality dictionary. Positive readout results remain conditional on this dictionary audit.",
    }
    RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {RESULT}")
    print(json.dumps(payload["audit"]["validation"], indent=2))


if __name__ == "__main__":
    main()
