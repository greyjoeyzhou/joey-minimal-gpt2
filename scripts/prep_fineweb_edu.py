"""Download and tokenize FineWeb-Edu 10B sample.

This is intended to be run on the 5090 box, not the Mac. It processes ~10B
tokens of educational web text and writes ~100 shards of 100M tokens each to
`data/edu_fineweb10B/`. Total disk footprint ~20 GB.

Architecture:
    - HuggingFace `datasets` streams the data in row-by-row.
    - A multiprocessing.Pool tokenizes documents in parallel (CPU-bound).
    - The main process accumulates tokens into a single buffer; when the
      buffer reaches `shard_size`, it's written to disk as a uint16 .bin.
    - Shard 0 is reserved for validation. Subsequent shards are training.

Run:
    uv run python scripts/prep_fineweb_edu.py
    # Optional: uv run python scripts/prep_fineweb_edu.py --shard_size 100_000_000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = Path("data/edu_fineweb10B")
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SHARD_SIZE_DEFAULT = 100_000_000  # 100M tokens per shard

# tiktoken's GPT-2 BPE encoding. Special tokens:
#   <|endoftext|> = 50256 (EOT). We use it to separate documents.
_enc = tiktoken.get_encoding("gpt2")
_EOT = _enc._special_tokens["<|endoftext|>"]  # 50256


def _tokenize(doc: dict) -> np.ndarray:
    """Tokenize one doc (dict from HF), prepend EOT.

    Returning np.uint16 directly keeps the shard write cheap. We assert tokens
    fit in uint16, which is true for GPT-2 vocab (50257 < 65535).
    """
    text = doc["text"]
    tokens = [_EOT]
    tokens.extend(_enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.uint32)
    assert (tokens_np < 2**16).all(), "Token id out of uint16 range — vocab mismatch."
    return tokens_np.astype(np.uint16)


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    tokens.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE_DEFAULT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stream the dataset. `split="train"` is HF's name for the whole sample.
    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG})...")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train")

    # Use ~half the CPU cores for tokenization to leave room for I/O and main.
    n_procs = max(1, (os.cpu_count() or 4) // 2)
    print(f"Tokenizing with {n_procs} workers, shard size = {args.shard_size:,} tokens")

    with mp.Pool(n_procs) as pool:
        shard_idx = 0
        # Preallocate the shard buffer once; copy into it.
        shard = np.empty(args.shard_size, dtype=np.uint16)
        pos = 0
        progress = tqdm(total=args.shard_size, unit="tok", desc=f"shard {shard_idx}")

        for tokens in pool.imap(_tokenize, ds, chunksize=16):
            # If this doc fits in the current shard, copy it in.
            if pos + len(tokens) < args.shard_size:
                shard[pos : pos + len(tokens)] = tokens
                pos += len(tokens)
                progress.update(len(tokens))
            else:
                # Flush the current shard. Take the prefix that fits, write,
                # then start a new shard with whatever's left.
                remaining = args.shard_size - pos
                shard[pos : pos + remaining] = tokens[:remaining]
                split = "val" if shard_idx == 0 else "train"
                out_path = args.out_dir / f"edufineweb_{split}_{shard_idx:06d}.bin"
                _write_shard(out_path, shard)
                progress.close()
                print(f"Wrote {out_path}")

                shard_idx += 1
                progress = tqdm(total=args.shard_size, unit="tok", desc=f"shard {shard_idx}")
                # Start the new shard with the leftover tokens.
                leftover = tokens[remaining:]
                shard[: len(leftover)] = leftover
                pos = len(leftover)
                progress.update(pos)

        # Flush any partial final shard (less than shard_size).
        if pos > 0:
            split = "val" if shard_idx == 0 else "train"
            out_path = args.out_dir / f"edufineweb_{split}_{shard_idx:06d}.bin"
            _write_shard(out_path, shard[:pos])
            progress.close()
            print(f"Wrote {out_path} (partial, {pos:,} tokens)")

    print(f"Done. {shard_idx + 1} shards written to {args.out_dir}")


if __name__ == "__main__":
    main()
