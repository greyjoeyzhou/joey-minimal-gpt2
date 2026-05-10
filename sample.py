"""Generate text from a trained checkpoint.

Usage:
    uv run python sample.py --ckpt checkpoints/model_005000.pt \
        --prompt "Hello, I'm a language model," \
        --max_tokens 128 --n_samples 3 --temperature 0.8 --top_k 50

Notes:
    - At checkpoint creation time we save raw_model.state_dict() (not the
      torch.compile wrapper), so we can load it cleanly without referencing
      torch.compile internals.
    - GPT-2 BPE tokenization is via tiktoken("gpt2").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from config import GPTConfig
from model import GPT
from utils import detect_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = detect_device()

    # Load checkpoint.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # We saved `config` (TrainConfig) but not GPTConfig — the model arch is
    # the default 124M, so we use GPTConfig() here. If you ever vary arch,
    # also save GPTConfig and load it here.
    model = GPT(GPTConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Tokenize prompt.
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(args.prompt)
    # Repeat the prompt across n_samples so they all share context, but
    # different RNG draws yield different completions.
    x = torch.tensor([prompt_ids] * args.n_samples, dtype=torch.long, device=device)

    # Generate.
    out = model.generate(
        x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    )

    # Decode and print.
    for i in range(args.n_samples):
        text = enc.decode(out[i].tolist())
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
