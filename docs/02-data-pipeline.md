# Data Pipeline: From FineWeb-Edu to Token Shards

This document explains how raw text becomes the (x, y) tensors the training
loop sees.

## The dataset: FineWeb-Edu (10B sample)

FineWeb-Edu is HuggingFace's curated subset of Common Crawl filtered for
educational content. It's the same dataset Karpathy's "Reproducing GPT-2"
video uses. The `sample-10BT` configuration is ~10B tokens (~50 GB raw text).

Why this and not OpenWebText / The Pile / C4?

- Higher quality on average (the "edu" filter removes a lot of low-signal web
  spam).
- Modern (2024), so it reflects current best practice.
- Available as a clean HuggingFace `datasets` stream — no manual reddit/web
  scraping.

## Tokenization: GPT-2 BPE (via tiktoken)

We use `tiktoken.get_encoding("gpt2")`. This is the same 50257-token BPE
vocabulary the original GPT-2 paper used. Same tokenizer means our token
counts and loss curves are directly comparable to GPT-2 in literature.

Each document is tokenized independently, then concatenated with an
`<|endoftext|>` (id 50256) separator. This signals "the previous document
ended" and lets the model learn document boundaries.

We pad the vocab from 50257 to 50304 in the *model* (not the tokenizer) for
matmul efficiency on tensor cores. The extra 47 rows in the embedding table
are never produced by tiktoken, so they contribute zero gradient.

## Sharding: uint16 .bin files

`scripts/prep_fineweb_edu.py` writes shards of 100M tokens each, as raw
`uint16` arrays on disk. Naming:

    data/edu_fineweb10B/
        edufineweb_val_000000.bin     # the first 100M tokens, reserved for val
        edufineweb_train_000001.bin
        edufineweb_train_000002.bin
        ...
        edufineweb_train_000099.bin   # ~last shard

Why uint16?

- 50304 < 65536, so it fits.
- Halves disk usage vs uint32.
- No header, no metadata. The format is "interpret this file as an array of
  uint16s." Loaders use `np.fromfile` or `np.memmap`.

Total disk: ~10B tokens × 2 bytes = ~20 GB.

Why not parquet / arrow / safetensors?

- Speed: `np.fromfile` of a flat uint16 array is the fastest possible read
  path. No deserialization overhead.
- Simplicity: you can `xxd` a shard and immediately understand the bytes.
- Resumability: shards are independent; if the prep script crashes, you
  re-run it (HF's cache is resumable, the script writes shard-by-shard).

## The loader: `data.py::DataLoaderLite`

Given a shard, the loader produces (x, y) windows of shape (B, T) where:

- `x[i, t]` = token at position t.
- `y[i, t]` = the next token, i.e., token at position t+1.

So `loss = cross_entropy(model(x), y)` trains the model to predict each next
token from the prefix.

### Window advancement

Each `next_batch()` advances by `B * T * world_size`. With `world_size=1`,
batches are contiguous. With DDP (`world_size > 1`), rank `r` starts at
offset `r*B*T` and skips ahead in lockstep — so the union of ranks reads a
disjoint stripe through each shard.

### Shard rollover

When a shard is exhausted, the loader rolls to the next shard. This is a
"round-robin within a split" loop, not a true epoch — for a 10B-token,
19k-step run we won't see the entire dataset, so per-epoch shuffling
doesn't matter.

## Working with the prep scripts

### Mac (development)

Use `scripts/prep_shakespeare.py` only. It downloads ~1 MB of tinyshakespeare
and writes `data/shakespeare/{train,val}.bin`. This is enough to smoke-test
the entire training loop.

### 5090 workstation (real training)

Run `scripts/prep_fineweb_edu.py` once. Expect:
- A few hours of wall time (CPU tokenization is the bottleneck).
- ~50 GB transient download from HF (cached at `~/.cache/huggingface/`).
- ~20 GB persistent on disk under `data/edu_fineweb10B/`.

You can blow away `~/.cache/huggingface/` after prep is done if you need disk
back.

## Verifying a shard

Roundtrip:

```python
import numpy as np, tiktoken
toks = np.fromfile("data/edu_fineweb10B/edufineweb_train_000001.bin", dtype=np.uint16)
enc = tiktoken.get_encoding("gpt2")
print(enc.decode(toks[:200].tolist()))
```
