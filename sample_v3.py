"""Generate text from a v3 checkpoint (MLA + MoE + Hyper-Connections + MTP).

Usage:
    uv run python sample_v3.py --ckpt checkpoints_v3/model_005000.pt \
        --prompt "Once upon a time" \
        --max_tokens 200 --n_samples 3 --temperature 0.8 --top_k 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from model_v3 import GPT
from utils import detect_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a model_v3 checkpoint")
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

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    print(f"Config: n_layer={model_cfg.n_layer}, n_head={model_cfg.n_head}, "
          f"n_kv_head={model_cfg.n_kv_head}, n_embd={model_cfg.n_embd}, "
          f"experts={model_cfg.n_shared_experts}s+{model_cfg.n_routed_experts}r "
          f"(top-{model_cfg.n_experts_per_tok}), hc={model_cfg.hc_expansion}")

    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(args.prompt)
    x = torch.tensor([prompt_ids] * args.n_samples, dtype=torch.long, device=device)

    out = model.generate(x, max_new_tokens=args.max_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    for i in range(args.n_samples):
        print(f"--- sample {i} ---")
        print(enc.decode(out[i].tolist()))
        print()


if __name__ == "__main__":
    main()
