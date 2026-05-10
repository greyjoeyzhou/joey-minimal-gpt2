"""Utility helpers shared across training, eval, and sampling.

Kept intentionally tiny and dependency-free (numpy, torch, stdlib only).
"""
from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def detect_device() -> str:
    """Pick the best available device in order: cuda > mps > cpu.

    Returns the *device type string* ('cuda', 'mps', 'cpu') rather than a
    torch.device, because some torch APIs (autocast, AdamW fused) want the
    string form.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs.

    This makes runs reproducible. Note: full determinism on CUDA also requires
    setting `torch.use_deterministic_algorithms(True)` and CUBLAS env vars,
    which trade speed for exactness. We don't enable that — for training, we
    just want the *initial* state to be reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set PYTHONHASHSEED for dict ordering reproducibility in any subprocesses.
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Cosine LR schedule with linear warmup.

    Three phases:
      1. Steps 0 .. warmup_steps-1: linear ramp from ~0 to max_lr.
         (Specifically: lr = max_lr * (step + 1) / warmup_steps.)
      2. Steps warmup_steps .. max_steps: cosine decay from max_lr to min_lr.
      3. Steps > max_steps: stays at min_lr.

    The "+1" in warmup means step 0 still has a tiny but nonzero LR. This is
    intentional: the very first batch never gets a 0-learning-rate update.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay from max_lr -> min_lr over (max_steps - warmup_steps) steps.
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    # cos(0) = 1, cos(pi) = -1. coeff goes from 1 -> 0 over the decay.
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class CSVLogger:
    """Append-only CSV logger for training metrics.

    The CSV has a unified schema where each row is one event. The `kind`
    column tells you what kind of event it is — train, val, or hella. Only
    the columns relevant to that kind are filled (others are empty strings).

    This single-table format makes it trivial to load later:
        df = pd.read_csv("logs/train.csv")
        df[df["kind"] == "train"].plot(x="step", y="loss")
    """

    FIELDS = [
        "step",
        "kind",  # train / val / hella
        "loss",
        "val_loss",
        "hella_acc",
        "lr",
        "dt_ms",
        "tokens_per_sec",
        "grad_norm",
    ]

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # If the file doesn't exist, create it and write the header.
        write_header = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def _write_row(self, row: dict[str, Any]) -> None:
        # Fill missing fields with empty string for readability.
        full_row = {k: row.get(k, "") for k in self.FIELDS}
        self._writer.writerow(full_row)
        self._file.flush()  # tail -f and crash-resilience matter more than throughput

    def log_train(
        self,
        step: int,
        loss: float,
        lr: float,
        dt_ms: float,
        tokens_per_sec: float,
        grad_norm: float,
    ) -> None:
        self._write_row(
            dict(
                step=step,
                kind="train",
                loss=loss,
                lr=lr,
                dt_ms=dt_ms,
                tokens_per_sec=tokens_per_sec,
                grad_norm=grad_norm,
            )
        )

    def log_val(self, step: int, val_loss: float) -> None:
        self._write_row(dict(step=step, kind="val", val_loss=val_loss))

    def log_hella(self, step: int, hella_acc: float) -> None:
        self._write_row(dict(step=step, kind="hella", hella_acc=hella_acc))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CSVLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
