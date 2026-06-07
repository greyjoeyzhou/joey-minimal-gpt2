"""Speed benchmark: compare step time and tokens/sec across model versions.

Uses synthetic random token data — no data directory required.

Usage:
    uv run python bench.py                     # benchmark v2
    uv run python bench.py --model v1          # benchmark v1
    uv run python bench.py --model v1 v2 v3   # compare all three
    uv run python bench.py --no-compile        # skip torch.compile

Output (per model, per step after warmup):
    step time (ms), tokens/sec, loss
    Summary: mean ± std over timed steps.
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore", message="Online softmax is disabled")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _load_model(name: str, device: str) -> torch.nn.Module:
    if name == "v1":
        from config_v1 import GPTConfig
        from model_v1 import GPT
    elif name == "v2":
        from config_v2 import GPTConfig
        from model_v2 import GPT
    elif name == "v3":
        from config_v3 import GPTConfig
        from model_v3 import GPT
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose v1, v2, or v3.")
    return GPT(GPTConfig()).to(device)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    model_name: str,
    device: str,
    micro_batch: int,
    seq_len: int,
    warmup: int,
    steps: int,
    compile: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}   device: {device}   compile: {compile}")
    print(f"  micro_batch={micro_batch}  seq_len={seq_len}  "
          f"tokens/step={micro_batch * seq_len:,}")
    print(f"{'='*60}")

    model = _load_model(model_name, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f}M")

    if compile and device == "cuda":
        print("  Compiling...")
        model = torch.compile(model)

    optimizer = (model._orig_mod if hasattr(model, "_orig_mod") else model).configure_optimizers(
        weight_decay=0.1, learning_rate=3e-4, device_type=device
    )

    vocab_size = 50304
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def one_step() -> float:
        x = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
        y = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.item())

    # Warmup — lets torch.compile finish and CUDA warmup kernels
    print(f"  Warming up ({warmup} steps)...")
    for i in range(warmup):
        loss = one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"    warmup {i+1}/{warmup}  loss={loss:.4f}")

    # Timed steps
    times: list[float] = []
    tokens_per_sec: list[float] = []
    print(f"  Timing ({steps} steps)...")
    for i in range(steps):
        t0 = time.perf_counter()
        loss = one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tps = (micro_batch * seq_len) / dt
        times.append(dt * 1000)
        tokens_per_sec.append(tps)
        print(f"    step {i+1:2d}  dt={dt*1000:7.1f}ms  tok/s={tps:,.0f}  loss={loss:.4f}")

    import statistics
    mean_dt  = statistics.mean(times)
    std_dt   = statistics.stdev(times) if len(times) > 1 else 0.0
    mean_tps = statistics.mean(tokens_per_sec)
    print(f"\n  Summary [{model_name}]: "
          f"dt = {mean_dt:.1f} ± {std_dt:.1f} ms | "
          f"tok/s = {mean_tps:,.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Speed benchmark across model versions")
    parser.add_argument("--model",       nargs="+", default=["v2"],
                        help="Model(s) to benchmark: v1, v2, v3 (default: v2)")
    parser.add_argument("--micro-batch", type=int, default=16,
                        help="Micro-batch size (default: 16, matches training config)")
    parser.add_argument("--seq-len",     type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--warmup",      type=int, default=5,
                        help="Warmup steps before timing (default: 5)")
    parser.add_argument("--steps",       type=int, default=10,
                        help="Timed steps (default: 10)")
    parser.add_argument("--compile",     action="store_true", default=True)
    parser.add_argument("--no-compile",  dest="compile", action="store_false")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for model_name in args.model:
        benchmark(
            model_name=model_name,
            device=device,
            micro_batch=args.micro_batch,
            seq_len=args.seq_len,
            warmup=args.warmup,
            steps=args.steps,
            compile=args.compile,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
