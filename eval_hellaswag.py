"""Zero-shot HellaSwag evaluation.

HellaSwag (Zellers et al. 2019) is a multi-choice commonsense benchmark.
Each example has:
  - ctx: a context sentence.
  - endings: 4 candidate continuations.
  - label: the correct continuation index.

Scoring:
    For each (ctx, ending) pair, we form ctx + ending, run it through the
    model, and compute the average per-token cross-entropy on the *ending
    tokens only* (we don't include context loss). Lowest NLL wins.

This is the standard "completion" zero-shot eval. GPT-2 124M baseline ~28-29%;
random is 25%.

Usage as a module:
    from eval_hellaswag import evaluate_hellaswag
    acc = evaluate_hellaswag(model, device, max_examples=1000)

Standalone:
    uv run python eval_hellaswag.py --ckpt checkpoints/model_005000.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import GPTConfig
from model import GPT
from utils import detect_device

URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
LOCAL = Path("data/hellaswag/hellaswag_val.jsonl")


def _download_if_needed() -> Path:
    if LOCAL.exists():
        return LOCAL
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    LOCAL.write_bytes(r.content)
    return LOCAL


def _render_example(example: dict, enc) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Render one example as (tokens, mask, label).

    - tokens: shape (4, T) — context+ending for each of 4 choices, right-padded.
    - mask:   shape (4, T) — 1 where ending tokens are, 0 for context or pad.
    - label:  int — correct ending index.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_ids = enc.encode_ordinary(ctx)
    rows = []
    masks = []
    for ending in endings:
        # Endings begin with a leading space in HellaSwag. Encode as part of
        # the same string flow so BPE works correctly.
        end_ids = enc.encode_ordinary(" " + ending)
        row = ctx_ids + end_ids
        mask = [0] * len(ctx_ids) + [1] * len(end_ids)
        rows.append(row)
        masks.append(mask)

    # Pad to the max length in this 4-way set with EOT (50256). Mask stays 0
    # on padding so they don't contribute to the NLL.
    max_len = max(len(r) for r in rows)
    PAD = 50256
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks, strict=True)):
        tokens[i, : len(r)] = torch.tensor(r)
        tokens[i, len(r) :] = PAD
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, label


@torch.no_grad()
def evaluate_hellaswag(
    model: torch.nn.Module,
    device: str,
    max_examples: int | None = None,
) -> float:
    """Run HellaSwag on the model. Returns accuracy in [0, 1]."""
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    path = _download_if_needed()

    correct = 0
    total = 0
    with open(path) as f:
        for line_idx, line in enumerate(tqdm(f, desc="hellaswag")):
            if max_examples is not None and line_idx >= max_examples:
                break
            example = json.loads(line)
            tokens, mask, label = _render_example(example, enc)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # Forward pass on all 4 candidates as one batch.
            autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
                logits, _ = model(tokens)
            # Shift: logit at position i predicts token at position i+1.
            shift_logits = logits[..., :-1, :].contiguous()  # (4, T-1, V)
            shift_targets = tokens[..., 1:].contiguous()  # (4, T-1)
            shift_mask = mask[..., 1:].contiguous().float()  # (4, T-1)

            # Per-token loss (no reduction).
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(
                shift_targets.size()
            )
            # Masked sum / masked count = mean loss on ending tokens only.
            sum_loss = (loss * shift_mask).sum(dim=1)
            n_tokens = shift_mask.sum(dim=1).clamp(min=1)
            avg_loss = sum_loss / n_tokens  # (4,)

            pred = int(avg_loss.argmin().item())
            if pred == label:
                correct += 1
            total += 1

    model.train()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    device = detect_device()
    model = GPT(GPTConfig()).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    acc = evaluate_hellaswag(model, device, max_examples=args.max_examples)
    print(f"HellaSwag accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
