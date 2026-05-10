# Training Recipe

This is the "what knobs and why" for `train.py`.

## Optimizer: AdamW

```
AdamW(params, lr=schedule(step), betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
```

- `betas=(0.9, 0.95)`: GPT-2 paper. The default torch AdamW uses 0.999 for
  beta2; 0.95 makes the second-moment estimate more responsive, which seems
  to matter for language modeling.
- `weight_decay=0.1`: applied only to 2D+ params (Linear weights, embeddings).
  1D params (biases, LN params) get no decay. This is *decoupled* weight
  decay (the W in AdamW): the decay is applied separately from the gradient
  step, not folded into the gradient.
- `fused=True` (CUDA only): one CUDA kernel for the whole optimizer step.

## Learning rate schedule

```
LR(step):
  if step < warmup_steps:          lr = max_lr * (step+1) / warmup_steps
  elif step >= max_steps:          lr = min_lr
  else:                            lr = min_lr + 0.5 * (1 + cos(pi * decay_ratio)) * (max_lr - min_lr)
```

Three phases: linear warmup, cosine decay, flat at min_lr.

Numbers:
- max_lr = 6e-4
- min_lr = 6e-5 (10% of max)
- warmup_steps = 715 (~ 375M tokens of warmup, matching GPT-3 paper at this scale)
- max_steps = 19_073 (~ 10B tokens at 524288 tokens/step)

Why cosine? Empirically it's a slightly smoother LR ramp-down than linear and
gives a small quality boost. Why not constant-then-decay? Cosine has been the
default for transformer pretraining since GPT-2.

## Token budget per step

The "magic number" is 524288 tokens per optimizer step. This is 2^19, the
batch the GPT-2 paper used. We hit it via gradient accumulation:

    tokens_per_step = micro_batch_size × seq_len × grad_accum_steps

On a single 5090, expect `seq_len=1024` and you'll tune `micro_batch_size`
and `grad_accum_steps` to multiply to 524288.

Starting guess: B=32, T=1024, grad_accum=16. If OOM, B=16/accum=32, etc.

## Gradient accumulation

Instead of running one giant batch, we run `grad_accum_steps` smaller batches
and accumulate gradients before stepping the optimizer:

```python
optimizer.zero_grad()
for _ in range(grad_accum_steps):
    x, y = loader.next_batch()
    loss = model(x, y)
    (loss / grad_accum_steps).backward()  # accumulate scaled gradients
optimizer.step()
```

The division by `grad_accum_steps` is so that `backward()` accumulates a
*mean* gradient (not a sum). The gradient that gets passed to optimizer.step()
is mathematically equivalent to one big batch of size `B * grad_accum_steps`.

## Mixed precision (bfloat16)

We wrap the forward pass in `torch.autocast(dtype=torch.bfloat16)`:

- Activations and intermediate computations are bf16.
- Optimizer states (Adam m, v) remain fp32.
- Gradients accumulated in fp32.

bf16 (vs fp16) has the same exponent range as fp32 (8 bits), so we don't need
loss scaling. The 5090 (Blackwell) has fast bf16 tensor cores.

Memory and speed math:
- bf16 activations halve the activation memory vs fp32.
- Tensor cores at bf16 are ~2x throughput vs fp32 on Blackwell.
- Net: ~1.5-2x speedup, half the activation memory.

## `torch.compile`

We wrap the model: `model = torch.compile(model)`. PyTorch's TorchInductor
JIT-compiles the model graph into fused CUDA kernels. First step is slow
(compiling); subsequent steps are 1.5-2x faster than eager mode.

We access the original (uncompiled) module via `model._orig_mod` whenever we
need to call non-forward methods like `configure_optimizers`.

## Gradient clipping

```
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Computes the global L2 norm of all gradients; if it exceeds 1.0, scales
everything down proportionally. Prevents single-step gradient spikes from
destabilizing training.

We log the grad norm at every step — sudden jumps are an early warning of
divergence.

## What we don't tune (intentionally)

Just to repeat the design's "out of scope" list with the recipe context:

- No DDP / FSDP / ZeRO — model fits on one 5090.
- No LR finder / hyperparam search — we use GPT-2's published recipe.
- No EMA, no SWA, no model averaging.
- No grad accumulation across multiple optimizer "passes" (more than 1
  optimizer step per gradient batch).
- No 8-bit AdamW / no optimizer offloading — fits in VRAM.

## Reading the training output

Each row in `logs/train.csv` is one of:

- `kind=train`: per-step loss, lr, dt, tokens/sec, grad norm.
- `kind=val`: averaged val loss at a `val_interval` boundary.
- `kind=hella`: HellaSwag accuracy at a `hella_interval` boundary.

To plot:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("logs/train.csv")
train = df[df["kind"] == "train"]
plt.plot(train["step"], train["loss"]); plt.yscale("log")
```
