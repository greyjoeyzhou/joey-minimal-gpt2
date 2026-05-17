"""Generate text from a v1 checkpoint (modern architecture).

Differences from sample.py:
  - Loads GPTConfig from the checkpoint (saved by train_v1.py) instead of
    assuming the default 124M config. This is important because v1 has
    non-default fields (n_kv_head, rope_theta, block_size).
  - Imports model_v1.GPT instead of model.GPT.

Usage:
    uv run python sample_v1.py --ckpt checkpoints_v1/model_005000.pt \
        --prompt "Hello, I'm a language model," \
        --max_tokens 128 --n_samples 3 --temperature 0.8 --top_k 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from model_v1 import GPT
from utils import detect_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a model_v1 checkpoint")
    p.add_argument("--ckpt",        type=Path,  required=True)
    p.add_argument("--prompt",      type=str,   default="Hello, I'm a language model,")
    p.add_argument("--max_tokens",  type=int,   default=128)
    p.add_argument("--n_samples",   type=int,   default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = detect_device()

    # Load checkpoint. train_v1.py saves "model_config" (a GPTConfig dataclass)
    # alongside the model weights. We use it to reconstruct the exact architecture.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    print(f"Loaded config: n_layer={model_cfg.n_layer}, n_head={model_cfg.n_head}, "
          f"n_kv_head={model_cfg.n_kv_head}, n_embd={model_cfg.n_embd}, "
          f"block_size={model_cfg.block_size}")

    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Tokenize with the same GPT-2 BPE tokenizer (unchanged from v0).
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(args.prompt)
    x = torch.tensor([prompt_ids] * args.n_samples, dtype=torch.long, device=device)

    out = model.generate(
        x,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    for i in range(args.n_samples):
        text = enc.decode(out[i].tolist())
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
