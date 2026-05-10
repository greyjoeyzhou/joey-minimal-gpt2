"""Tests for utils.py — lr schedule, device, csv logger, seeding."""
import csv
from pathlib import Path

import pytest
import torch

from utils import (
    CSVLogger,
    detect_device,
    get_lr,
    seed_everything,
)


def test_get_lr_warmup_phase():
    """During warmup, LR ramps linearly from 0 to max_lr."""
    max_lr = 6e-4
    warmup = 100
    max_steps = 1000
    min_lr = 6e-5
    # Step 0 (boundary): tiny but positive.
    assert get_lr(0, max_lr, min_lr, warmup, max_steps) == pytest.approx(max_lr / warmup)
    # Step warmup-1: very close to max_lr.
    assert get_lr(warmup - 1, max_lr, min_lr, warmup, max_steps) == pytest.approx(max_lr)


def test_get_lr_cosine_decay():
    """After warmup, LR cosine-decays from max_lr down to min_lr."""
    max_lr = 6e-4
    min_lr = 6e-5
    warmup = 100
    max_steps = 1000
    # Just after warmup: ~ max_lr.
    assert get_lr(warmup, max_lr, min_lr, warmup, max_steps) == pytest.approx(max_lr, rel=1e-3)
    # At the end: min_lr.
    assert get_lr(max_steps, max_lr, min_lr, warmup, max_steps) == pytest.approx(min_lr)
    # Past the end: stays at min_lr.
    assert get_lr(max_steps + 500, max_lr, min_lr, warmup, max_steps) == pytest.approx(min_lr)
    # Midpoint of cosine: between min_lr and max_lr.
    mid = get_lr((warmup + max_steps) // 2, max_lr, min_lr, warmup, max_steps)
    assert min_lr < mid < max_lr


def test_detect_device_returns_string():
    """detect_device should return one of 'cuda', 'mps', 'cpu'."""
    device = detect_device()
    assert device in ("cuda", "mps", "cpu")


def test_seed_everything_makes_torch_deterministic():
    """After seeding, two identical torch.randn calls should match."""
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_csv_logger_writes_rows(tmp_path: Path):
    """CSVLogger should create a file with the expected columns and rows."""
    log_path = tmp_path / "train.csv"
    logger = CSVLogger(log_path)
    logger.log_train(step=0, loss=4.5, lr=1e-5, dt_ms=300.0, tokens_per_sec=50_000.0, grad_norm=1.2)
    logger.log_val(step=100, val_loss=3.9)
    logger.log_hella(step=500, hella_acc=0.27)
    logger.close()

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["kind"] == "train"
    assert rows[0]["loss"] == "4.5"
    assert rows[1]["kind"] == "val"
    assert rows[1]["val_loss"] == "3.9"
    assert rows[2]["kind"] == "hella"
    assert rows[2]["hella_acc"] == "0.27"
