"""HellaSwag evaluation for model_v3.

Identical scoring logic to eval_hellaswag_v1.py. Loads GPTConfig from
the checkpoint saved by train_v3.py.

Usage:
    uv run python eval_hellaswag_v3.py --ckpt checkpoints_v3/model_005000.pt
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

from model_v3 import GPT
from utils import detect_device

URL   = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
LOCAL = Path("data/hellaswag/hellaswag_val.jsonl")


def _download_if_needed() -> Path:
    if LOCAL.exists():
        return LOCAL
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    LOCAL.write_bytes(r.content)
    return LOCAL


def _render_example(example: dict, enc) -> tuple[torch.Tensor, torch.Tensor, int]:
    ctx, label, endings = example["ctx"], example["label"], example["endings"]
    ctx_ids = enc.encode_ordinary(ctx)
    rows, masks = [], []
    for ending in endings:
        end_ids = enc.encode_ordinary(" " + ending)
        rows.append(ctx_ids + end_ids)
        masks.append([0] * len(ctx_ids) + [1] * len(end_ids))
    max_len = max(len(r) for r in rows)
    PAD = 50256
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask   = torch.zeros((4, max_len), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks, strict=True)):
        tokens[i, :len(r)] = torch.tensor(r)
        tokens[i, len(r):] = PAD
        mask[i,   :len(m)] = torch.tensor(m)
    return tokens, mask, label


@torch.no_grad()
def evaluate_hellaswag(
    model: torch.nn.Module,
    device: str,
    max_examples: int | None = None,
) -> float:
    model.eval()
    enc  = tiktoken.get_encoding("gpt2")
    path = _download_if_needed()
    correct = total = 0
    with open(path) as f:
        for line_idx, line in enumerate(tqdm(f, desc="hellaswag")):
            if max_examples is not None and line_idx >= max_examples:
                break
            example = json.loads(line)
            tokens, mask, label = _render_example(example, enc)
            tokens, mask = tokens.to(device), mask.to(device)
            autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
                logits, _ = model(tokens)
            shift_logits  = logits[..., :-1, :].contiguous()
            shift_targets = tokens[..., 1:].contiguous()
            shift_mask    = mask[..., 1:].contiguous().float()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction="none",
            ).view(shift_targets.size())
            avg_loss = (loss * shift_mask).sum(1) / shift_mask.sum(1).clamp(min=1)
            if int(avg_loss.argmin()) == label:
                correct += 1
            total += 1
    model.train()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         type=Path, required=True)
    parser.add_argument("--max_examples", type=int,  default=None)
    args = parser.parse_args()

    device = detect_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    print(f"Config: n_layer={model_cfg.n_layer}, n_head={model_cfg.n_head}, "
          f"n_kv_head={model_cfg.n_kv_head}, n_embd={model_cfg.n_embd}, "
          f"experts={model_cfg.n_shared_experts}s+{model_cfg.n_routed_experts}r")

    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    acc = evaluate_hellaswag(model, device, max_examples=args.max_examples)
    print(f"HellaSwag accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
