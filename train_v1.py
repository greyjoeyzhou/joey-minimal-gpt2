"""Training loop for model_v1 (modern architecture).

Structurally identical to train.py. The differences are:

  1. Imports model_v1.GPT and config_v1.{GPTConfig, TrainConfig}.
  2. GPTConfig is saved inside the checkpoint — critical because the v1
     config has non-default fields (n_kv_head, rope_theta, block_size) that
     must be known to reconstruct the model at eval/sample time. The GPT-2
     train.py didn't need this because GPTConfig() had only one valid config.
  3. seq_len defaults to 2048 (double the GPT-2 run's 1024).

Run:
    uv run python train_v1.py
With overrides:
    uv run python train_v1.py --micro_batch_size=8 --grad_accum_steps=32
Resume from checkpoint:
    uv run python train_v1.py --resume checkpoints_v1/model_005000.pt
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch

from config_v1 import GPTConfig, TrainConfig
from data import DataLoaderLite
from model_v1 import GPT
from utils import CSVLogger, detect_device, get_lr, seed_everything


def _build_argparser() -> argparse.ArgumentParser:
    """CLI auto-derived from TrainConfig fields. Edit defaults in config_v1.py."""
    parser = argparse.ArgumentParser(description="Train modern GPT v1 (RoPE+GQA+SwiGLU+RMSNorm)")
    defaults = TrainConfig()
    for f in fields(defaults):
        if f.type is Path:
            parser.add_argument(f"--{f.name}", type=str, default=str(getattr(defaults, f.name)))
        elif f.type is int:
            parser.add_argument(f"--{f.name}", type=int, default=getattr(defaults, f.name))
        elif f.type is float:
            parser.add_argument(f"--{f.name}", type=float, default=getattr(defaults, f.name))
        else:
            parser.add_argument(f"--{f.name}", default=getattr(defaults, f.name))
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to a v1 checkpoint to resume from. Empty = train from scratch.",
    )
    parser.add_argument("--compile",    action="store_true", default=True)
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

    Returns (loss_accum, grad_norm). Identical to train.py.
    """
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    for _ in range(grad_accum_steps):
        x, y = loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()
    return loss_accum, float(grad_norm.item())


def main() -> None:
    args = _build_argparser().parse_args()

    cfg_kwargs = {f.name: getattr(args, f.name) for f in fields(TrainConfig())}
    for k in ("log_dir", "ckpt_dir", "data_dir"):
        cfg_kwargs[k] = Path(cfg_kwargs[k])
    cfg = TrainConfig(**cfg_kwargs)

    seed_everything(cfg.seed)
    device = detect_device()
    print(f"Device: {device}")
    print(f"Tokens per step: {cfg.tokens_per_step:,}")
    print(f"Max steps: {cfg.max_steps:,}")
    print(f"Total tokens: {cfg.tokens_per_step * cfg.max_steps:,}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # --- Model ---
    model_cfg = GPTConfig()
    model = GPT(model_cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_head={model_cfg.n_head}, n_kv_head={model_cfg.n_kv_head}, "
          f"n_embd={model_cfg.n_embd}, n_layer={model_cfg.n_layer}, "
          f"block_size={model_cfg.block_size}")

    if args.compile and device == "cuda":
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)  # type: ignore[assignment]

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, device_type=device
    )

    # --- Data ---
    train_loader = DataLoaderLite(split="train", B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir)
    val_loader   = DataLoaderLite(split="val",   B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir)

    # --- Resume ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
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

        if step % cfg.val_interval == 0 and step > 0:
            val_loss = _evaluate(model, val_loader, cfg.val_iters, device)
            logger.log_val(step, val_loss)
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        if step % cfg.hella_interval == 0 and step > 0:
            from eval_hellaswag_v1 import evaluate_hellaswag
            acc = evaluate_hellaswag(raw_model, device)
            logger.log_hella(step, acc)
            print(f"step {step:6d} | hella_acc {acc:.4f}")

        if step % cfg.save_interval == 0 and step > 0:
            ckpt_path = cfg.ckpt_dir / f"model_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # Save GPTConfig so eval/sample scripts can reconstruct the
                    # exact architecture without guessing default fields.
                    "model_config": model_cfg,
                    "train_config": cfg,
                },
                ckpt_path,
            )
            print(f"step {step:6d} | saved {ckpt_path}")

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
            step=step, loss=loss_accum, lr=lr,
            dt_ms=dt * 1000, tokens_per_sec=tokens_per_sec, grad_norm=grad_norm,
        )
        if step < 20 or step % 10 == 0:
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                f"dt {dt*1000:6.1f}ms | tok/s {tokens_per_sec:,.0f}"
            )

    # Final checkpoint.
    final_step = cfg.max_steps - 1
    ckpt_path = cfg.ckpt_dir / f"model_{final_step:06d}.pt"
    torch.save(
        {
            "step": final_step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_cfg,
            "train_config": cfg,
        },
        ckpt_path,
    )
    print(f"step {final_step:6d} | saved {ckpt_path}")
    logger.close()
    print("Training complete.")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, val_loader: DataLoaderLite, iters: int, device: str) -> float:
    """Mean cross-entropy over `iters` validation batches."""
    model.eval()
    total = 0.0
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for _ in range(iters):
        x, y = val_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
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
    """Tiny in-process training run for smoke tests.

    Uses a tiny v1 model (2 layers, 4 Q heads, 2 KV heads, n_embd=64) on
    tinyshakespeare. Runs on Mac CPU/MPS in seconds.
    """
    seed_everything(1337)
    device = detect_device()

    cfg = GPTConfig(
        n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
        block_size=seq_len, vocab_size=50304,
    )
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-3, device_type=device)
    loader = DataLoaderLite(
        split="train", B=micro_batch_size, T=seq_len, data_dir=Path("data/shakespeare"),
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
