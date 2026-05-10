"""Tiny Shakespeare dataset prep.

Downloads tinyshakespeare (~1 MB of plain text), tokenizes it with the GPT-2
BPE tokenizer, and writes train/val shards as raw uint16 .bin files.

Used by `tests/test_smoke_train.py` and for any local Mac smoke testing.
The full FineWeb-Edu pipeline is `scripts/prep_fineweb_edu.py`.

Run:
    uv run python scripts/prep_shakespeare.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests
import tiktoken

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_DIR = Path("data/shakespeare")
TRAIN_FRAC = 0.9


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUT_DIR / "input.txt"
    if not raw_path.exists():
        print(f"Downloading {URL}...")
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        raw_path.write_text(r.text)
    text = raw_path.read_text()
    print(f"Read {len(text):,} characters")

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    # encode_ordinary skips special-token handling — we just want raw BPE.
    tokens_arr = np.array(tokens, dtype=np.uint16)
    print(f"Tokenized to {len(tokens_arr):,} tokens (vocab size 50257)")

    split_idx = int(len(tokens_arr) * TRAIN_FRAC)
    train = tokens_arr[:split_idx]
    val = tokens_arr[split_idx:]

    train_path = OUT_DIR / "train.bin"
    val_path = OUT_DIR / "val.bin"
    train.tofile(train_path)
    val.tofile(val_path)
    print(f"Wrote {train_path} ({train.size:,} tokens)")
    print(f"Wrote {val_path}   ({val.size:,} tokens)")


if __name__ == "__main__":
    main()
