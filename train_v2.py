"""Training loop for model_v2 (MLA + MoE + Hyper-Connections + MTP).

Identical structure to train_v1.py. Notable differences:

  - The MoE load-balance loss and MTP loss are embedded inside model.forward(),
    so the training loop does not need to handle them separately. They show up
    in the logged `loss` value. To see each component separately, add
    debug prints inside GPT.forward() temporarily.

  - Per-step wall-clock time will be ~30-50% longer than v1 due to MoE dispatch
    and the larger HC hidden state. Tokens/sec will be lower but the model
    should learn faster per token.

Run:
    uv run python train_v2.py
With overrides:
    uv run python train_v2.py --micro_batch_size=8 --grad_accum_steps=32
Resume:
    uv run python train_v2.py --resume checkpoints_v2/model_005000.pt
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch

from config_v2 import GPTConfig, TrainConfig
from data import DataLoaderLite
from model_v2 import GPT
from utils import CSVLogger, detect_device, get_lr, seed_everything


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GPT v2 (MLA+MoE+HC+MTP)")
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
    parser.add_argument("--resume",     type=str, default="")
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
    """One optimizer step = grad_accum_steps micro-steps. Returns (loss, grad_norm)."""
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
    print(f"Tokens per step: {cfg.tokens_per_step:,} | Max steps: {cfg.max_steps:,}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model_cfg = GPTConfig()
    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_active = sum(  # active params per token (shared + top-k routed experts only)
        p.numel() for n, p in model.named_parameters()
        if "routed_experts" not in n or any(
            f".routed_experts.{i}." in n for i in range(model_cfg.n_experts_per_tok)
        )
    )
    print(f"Total params:  {n_params:,}")
    print(f"Config: n_layer={model_cfg.n_layer}, n_head={model_cfg.n_head}, "
          f"n_kv_head={model_cfg.n_kv_head}, n_embd={model_cfg.n_embd}, "
          f"hc_expansion={model_cfg.hc_expansion}, "
          f"experts={model_cfg.n_shared_experts}shared+{model_cfg.n_routed_experts}routed"
          f"(top-{model_cfg.n_experts_per_tok})")

    if args.compile and device == "cuda":
        print("Compiling with torch.compile()...")
        model = torch.compile(model)  # type: ignore[assignment]

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, device_type=device
    )

    train_loader = DataLoaderLite(split="train", B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir)
    val_loader   = DataLoaderLite(split="val",   B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(cfg.log_dir / "train.csv")

    print("Starting training loop")
    for step in range(start_step, cfg.max_steps):

        if step % cfg.val_interval == 0 and step > 0:
            val_loss = _evaluate(model, val_loader, cfg.val_iters, device)
            logger.log_val(step, val_loss)
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        if step % cfg.hella_interval == 0 and step > 0:
            from eval_hellaswag_v2 import evaluate_hellaswag
            acc = evaluate_hellaswag(raw_model, device)
            logger.log_hella(step, acc)
            print(f"step {step:6d} | hella_acc {acc:.4f}")

        if step % cfg.save_interval == 0 and step > 0:
            ckpt_path = cfg.ckpt_dir / f"model_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_cfg,
                "train_config": cfg,
            }, ckpt_path)
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

    final_step = cfg.max_steps - 1
    ckpt_path = cfg.ckpt_dir / f"model_{final_step:06d}.pt"
    torch.save({
        "step": final_step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_cfg,
        "train_config": cfg,
    }, ckpt_path)
    print(f"step {final_step:6d} | saved {ckpt_path}")
    logger.close()
    print("Training complete.")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, val_loader: DataLoaderLite, iters: int, device: str) -> float:
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
    """Tiny smoke-test on tinyshakespeare. Runs on Mac CPU/MPS in ~60s."""
    seed_everything(1337)
    device = detect_device()
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_kv_head=1,
        n_embd=96,            # 4 heads * (8 rope + 16 nope) = 4 * 24 = 96
        rope_head_dim=8, nope_head_dim=16,
        q_lora_rank=32, kv_lora_rank=16,
        n_routed_experts=2, n_shared_experts=1, n_experts_per_tok=1,
        moe_intermediate=64, hc_expansion=2, n_mtp=1,
        block_size=seq_len, vocab_size=50304,
    )
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-3, device_type=device)
    loader = DataLoaderLite(
        split="train", B=micro_batch_size, T=seq_len, data_dir=Path("data/shakespeare"),
    )
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


if __name__ == "__main__":
    main()
