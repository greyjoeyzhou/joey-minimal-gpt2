"""Tests for the sharded data loader.

These rely on the tinyshakespeare data being prepared
(scripts/prep_shakespeare.py).
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from data import DataLoaderLite

SHAKE = Path("data/shakespeare")


def setup_module(module):
    """Skip the whole module if shakespeare data isn't prepared."""
    if not (SHAKE / "train.bin").exists():
        pytest.skip("Run scripts/prep_shakespeare.py first to generate test data.")


def test_loader_returns_correct_shapes():
    loader = DataLoaderLite(split="train", B=2, T=16, data_dir=SHAKE, shard_glob="*.bin")
    x, y = loader.next_batch()
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_loader_y_is_x_shifted_by_one():
    """For language modeling, y[i] should equal x[i+1] (the next-token target)."""
    loader = DataLoaderLite(split="train", B=1, T=32, data_dir=SHAKE, shard_glob="*.bin")
    x, y = loader.next_batch()
    # y[0, :-1] should equal x[0, 1:]
    assert torch.equal(y[0, :-1], x[0, 1:])


def test_loader_advances_position():
    """Consecutive next_batch() calls should return non-overlapping chunks."""
    loader = DataLoaderLite(split="train", B=1, T=16, data_dir=SHAKE, shard_glob="*.bin")
    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()
    assert not torch.equal(x1, x2)
    # In rank=0/world=1 setup, x2 should start where x1's targets ended.


def test_loader_rolls_over_to_next_shard():
    """When we exhaust a shard, the loader should roll forward (back to 0 if only one)."""
    # Use a tiny T so we definitely wrap.
    loader = DataLoaderLite(split="val", B=1, T=8, data_dir=SHAKE, shard_glob="*.bin")
    # The val shard has ~33k tokens. Read enough batches to exhaust it.
    val_tokens = np.fromfile(SHAKE / "val.bin", dtype=np.uint16).size
    n_batches = val_tokens // (1 * 8) + 5  # overshoot
    for _ in range(n_batches):
        x, y = loader.next_batch()
        assert x.shape == (1, 8)


def test_loader_respects_rank_world_size():
    """With world_size > 1, each rank should read a non-overlapping slice."""
    loader_a = DataLoaderLite(split="train", B=2, T=16, data_dir=SHAKE, shard_glob="*.bin", rank=0, world_size=2)
    loader_b = DataLoaderLite(split="train", B=2, T=16, data_dir=SHAKE, shard_glob="*.bin", rank=1, world_size=2)
    xa, _ = loader_a.next_batch()
    xb, _ = loader_b.next_batch()
    # Different ranks should get different data on the first call.
    assert not torch.equal(xa, xb)
