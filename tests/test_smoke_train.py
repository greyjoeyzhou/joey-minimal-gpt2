"""End-to-end smoke test of the training loop on tinyshakespeare.

Runs ~20 optimizer steps with a tiny model on the shakespeare shards. The
assertion is that the loss drops meaningfully. This is the canary that
catches *plumbing* bugs (wrong shapes, optimizer not stepping, loss not
flowing through autocast, etc.) before you push code to the 5090.

Runs in ~30 seconds on Mac MPS or CPU.
"""
from pathlib import Path

import pytest

from train import train_smoke

SHAKE = Path("data/shakespeare")


def test_smoke_train_loss_drops():
    """20 steps on shakespeare — final loss should be < 0.9 * initial."""
    if not (SHAKE / "train.bin").exists():
        pytest.skip("Run scripts/prep_shakespeare.py first.")
    losses = train_smoke(steps=20, micro_batch_size=4, seq_len=64, grad_accum_steps=1)
    assert losses[0] > losses[-1], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[-1] < losses[0] * 0.9, (
        f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
