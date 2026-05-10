"""GPT-2 124M training loop.

Two entry points:

- main(): the full training run. Reads CLI args, builds a TrainConfig, runs
  for cfg.max_steps. Writes CSV logs and checkpoints. Use this on the 5090.

- train_smoke(): a tiny in-process training run for the smoke test. Uses a
  toy model and the tinyshakespeare shards. Runs on Mac CPU/MPS in seconds.

The actual training step is identical in both paths; train_smoke just sets
small hyperparams and returns the loss curve for assertion.

Run on 5090:
    uv run python train.py
With overrides:
    uv run python train.py --micro_batch_size=16 --grad_accum_steps=32
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch

from config import GPTConfig, TrainConfig
from data import DataLoaderLite
from model import GPT
from utils import CSVLogger, detect_device, get_lr, seed_everything


def _build_argparser() -> argparse.ArgumentParser:
    """CLI auto-derived from TrainConfig fields. Edit defaults in config.py."""
    parser = argparse.ArgumentParser(description="Train GPT-2 124M")
    defaults = TrainConfig()
    for f in fields(defaults):
        # Path types: parse as strings, cast manually below.
        if f.type is Path:
            parser.add_argument(f"--{f.name}", type=str, default=str(getattr(defaults, f.name)))
        elif f.type is int:
            parser.add_argument(f"--{f.name}", type=int, default=getattr(defaults, f.name))
        elif f.type is float:
            parser.add_argument(f"--{f.name}", type=float, default=getattr(defaults, f.name))
        else:
            parser.add_argument(f"--{f.name}", default=getattr(defaults, f.name))
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint to resume from. Empty = train from scratch.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile (default True; pass --no-compile to disable).",
    )
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    return parser


def _train_one_step(
    model: torch.nn.Module,
    loader: DataLoaderLite,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accum_steps: int,
    grad_clip: float,
    lr: float,
) -> tuple[float, float]:
    """Execute one optimizer step (= grad_accum_steps micro-steps).

    Returns (loss_accum, grad_norm).
    """
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    # autocast dtype: bf16 on CUDA (no loss scaler needed), fp32 elsewhere.
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    for _ in range(grad_accum_steps):
        x, y = loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        # Scale loss so that backward() accumulates a *mean* gradient over the
        # grad_accum_steps micro-batches (not a sum).
        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Apply the LR set by the caller.
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()

    return loss_accum, float(grad_norm.item())


def main() -> None:
    args = _build_argparser().parse_args()

    # Build TrainConfig from CLI args. Manual Path casts where needed.
    cfg_kwargs = {f.name: getattr(args, f.name) for f in fields(TrainConfig())}
    for k in ("log_dir", "ckpt_dir", "data_dir"):
        cfg_kwargs[k] = Path(cfg_kwargs[k])
    cfg = TrainConfig(**cfg_kwargs)

    seed_everything(cfg.seed)
    device = detect_device()
    print(f"Device: {device}")
    print(f"Tokens per step: {cfg.tokens_per_step:,}")
    print(f"Max steps: {cfg.max_steps:,}")
    print(f"Total tokens to be trained: {cfg.tokens_per_step * cfg.max_steps:,}")

    # On CUDA, set float32 matmul precision to 'high' to allow TF32 on Ampere+.
    # On Blackwell (5090) this affects fp32 fallback paths; bf16 autocast is
    # the main precision regime.
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # --- Model ---
    model = GPT(GPTConfig()).to(device)
    if args.compile and device == "cuda":
        # torch.compile gives ~1.5-2x speedup on training. First step compiles,
        # so it'll appear slow; subsequent steps are fast.
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)  # type: ignore[assignment]

    # --- Optimizer ---
    # We access configure_optimizers via the original module even if compiled.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, device_type=device
    )

    # --- Data ---
    train_loader = DataLoaderLite(
        split="train", B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir
    )
    val_loader = DataLoaderLite(
        split="val", B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir
    )

    # --- Resume? ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    # --- Logger ---
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(cfg.log_dir / "train.csv")

    # --- Training loop ---
    print("Starting training loop")
    for step in range(start_step, cfg.max_steps):
        # Periodic eval (val loss).
        if step % cfg.val_interval == 0 and step > 0:
            val_loss = _evaluate(model, val_loader, cfg.val_iters, device)
            logger.log_val(step, val_loss)
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        # Periodic HellaSwag.
        if step % cfg.hella_interval == 0 and step > 0:
            from eval_hellaswag import evaluate_hellaswag  # lazy import

            acc = evaluate_hellaswag(raw_model, device)
            logger.log_hella(step, acc)
            print(f"step {step:6d} | hella_acc {acc:.4f}")

        # Periodic checkpoint.
        if step % cfg.save_interval == 0 and step > 0:
            ckpt_path = cfg.ckpt_dir / f"model_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"step {step:6d} | saved {ckpt_path}")

        # Training step.
        t0 = time.time()
        lr = get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        loss_accum, grad_norm = _train_one_step(
            model, train_loader, optimizer, device, cfg.grad_accum_steps, cfg.grad_clip, lr
        )
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = cfg.tokens_per_step / dt
        logger.log_train(
            step=step,
            loss=loss_accum,
            lr=lr,
            dt_ms=dt * 1000,
            tokens_per_sec=tokens_per_sec,
            grad_norm=grad_norm,
        )
        # Print every step early in training, then every 10 once we're settled.
        if step < 20 or step % 10 == 0:
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                f"dt {dt*1000:6.1f}ms | tok/s {tokens_per_sec:,.0f}"
            )

    logger.close()
    print("Training complete.")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, val_loader: DataLoaderLite, iters: int, device: str) -> float:
    """Mean cross-entropy over `iters` validation batches."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = val_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        total += float(loss.item())
    model.train()
    return total / iters


def train_smoke(
    steps: int = 20,
    micro_batch_size: int = 4,
    seq_len: int = 64,
    grad_accum_steps: int = 1,
) -> list[float]:
    """Tiny in-process training run on tinyshakespeare for the smoke test.

    Uses a *tiny* model (2 layers, 2 heads, n_embd=64) so it runs on a Mac
    CPU in seconds. Returns the per-step losses for assertion.
    """
    seed_everything(1337)
    device = detect_device()

    # Tiny model — not GPT-2 124M. We just want to verify the loop trains.
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=seq_len, vocab_size=50304)
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-3, device_type=device
    )

    loader = DataLoaderLite(
        split="train",
        B=micro_batch_size,
        T=seq_len,
        data_dir=Path("data/shakespeare"),
    )

    losses: list[float] = []
    for _ in range(steps):
        loss_accum, _ = _train_one_step(
            model, loader, optimizer, device, grad_accum_steps, grad_clip=1.0, lr=3e-3
        )
        losses.append(loss_accum)
    return losses


if __name__ == "__main__":
    main()
