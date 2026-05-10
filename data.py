"""On-disk sharded data loader for GPT pre-training.

Design:

- Each "shard" is a single .bin file of contiguous uint16 token IDs. No
  header, no compression — just np.fromfile or np.memmap to load.
- DataLoaderLite picks one shard at a time, hands out (x, y) batches from
  a sliding window, and rolls to the next shard when exhausted.
- For DDP, the loader is rank-aware: each call to next_batch advances by
  B*T*world_size, and rank `r` starts at offset r*B*T within each window.
  This way every rank sees a disjoint stripe of tokens.

This mirrors karpathy's DataLoaderLite in build-nanogpt almost exactly. Kept
"lite" because we deliberately skip prefetching, async loading, and shuffling.
For GPT-style training, sequential reads of large random-looking shards are
already cache-friendly enough.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _load_shard(path: Path) -> torch.Tensor:
    """Load a uint16 .bin shard into a torch.long tensor.

    We convert to long at load time (instead of every batch) because:
      - Embeddings expect int64 indices.
      - The conversion is cheap and happens once per shard.
      - Memory is fine: a 100M-token shard at int64 = 800 MB. We hold one at
        a time, and most shards won't be that big in dev settings. For very
        large shards you'd want memmap + per-batch cast — see the comments
        at the bottom of this file.
    """
    arr = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


class DataLoaderLite:
    """Stream batches of (x, y) from a directory of .bin token shards.

    Args:
        split: "train" or "val". Used to filter shard filenames.
        B: micro batch size.
        T: sequence length.
        data_dir: directory containing the shard files.
        shard_glob: glob pattern matching shards. E.g., "edufineweb_train_*.bin".
        rank: this process's rank in [0, world_size).
        world_size: total number of DDP processes.

    Conventions:
        - Shards are listed in sorted order.
        - For split filtering: shards whose name contains f"_{split}_" are picked.
          For tiny-shakespeare we also accept `{split}.bin` directly.
    """

    def __init__(
        self,
        split: str,
        B: int,
        T: int,
        data_dir: Path | str,
        shard_glob: str = "*.bin",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        assert split in {"train", "val"}
        assert 0 <= rank < world_size

        self.split = split
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size

        data_dir = Path(data_dir)
        all_shards = sorted(data_dir.glob(shard_glob))
        # Filter by split. Accept both "edufineweb_train_000001.bin" and "train.bin".
        shards = [
            s for s in all_shards
            if f"_{split}_" in s.name or s.stem == split
        ]
        if not shards:
            raise FileNotFoundError(
                f"No shards found for split='{split}' in {data_dir} "
                f"(glob={shard_glob!r}). Did you run the prep script?"
            )
        self.shards = shards
        self.reset()

    def reset(self) -> None:
        """Reset to the start of the first shard (start of an epoch)."""
        self.current_shard = 0
        self.tokens = _load_shard(self.shards[0])
        # rank `r` starts at offset r*B*T into the shard.
        self.current_position = self.B * self.T * self.rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (x, y) pair of shape (B, T).

        y is x shifted by one token, so cross_entropy(model(x), y) trains the
        model to predict each next token from the prefix.
        """
        B, T = self.B, self.T
        # We need B*T + 1 tokens (the +1 is so y can be shifted).
        end = self.current_position + B * T + 1
        buf = self.tokens[self.current_position : end]

        # If we overshot the shard, roll to the next one and retry from offset 0.
        if buf.size(0) < B * T + 1:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = _load_shard(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.rank
            end = self.current_position + B * T + 1
            buf = self.tokens[self.current_position : end]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance by B*T*world_size so adjacent ranks read non-overlapping stripes.
        self.current_position += B * T * self.world_size
        return x, y


# Notes for scaling up:
#
# 1. For very large shards (multi-GB), replace `_load_shard` with np.memmap
#    and slice + cast per batch. Tradeoff: per-batch overhead, but the OS
#    page cache handles random-ish reads well.
#
# 2. To do "true" epoch-aware training (shuffle shard order per epoch, etc.),
#    keep track of which shards you've seen and reset/permute when all are done.
#    For 10B-token / 19k-step runs, you won't even finish one epoch, so the
#    simple round-robin in reset() is fine.
#
# 3. For multi-machine DDP, you typically want the *same* shard list on every
#    rank, with the per-rank offset doing the actual splitting. That's what
#    we do here.
