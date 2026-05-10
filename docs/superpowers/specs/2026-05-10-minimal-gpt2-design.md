# Minimal GPT-2 Training Project — Design Spec

**Date:** 2026-05-10
**Status:** Approved — ready for implementation plan
**Reference:** Karpathy's `build-nanogpt` (2024 "Reproducing GPT-2" video and repo)

## 1. Goal & Scope

Build a minimal, heavily commented project that trains a GPT-2 124M model from scratch on ~10B tokens, for the purpose of learning the mechanics of modern transformer pre-training.

- **Model:** vanilla GPT-2 124M (12 layers, 12 heads, 768 dim, 1024 context, 50257 vocab). No architectural alterations versus nanoGPT.
- **Data:** FineWeb-Edu `sample-10BT` from HuggingFace, tokenized to on-disk `uint16` shards.
- **Eval:** validation loss on a held-out shard + zero-shot HellaSwag.
- **Workflow:** code/dev/smoke-test on Mac (no CUDA); data prep and training on a single Linux workstation with one RTX 5090 (32GB).
- **Docs:** heavy inline comments in code, plus 5 high-level markdown docs under `docs/`.

### Non-goals

- Architectural research (changing attention, positional encoding, MLP, etc.).
- Multi-GPU or multi-node training (kept structurally compatible for later; not wired up).
- Production-grade tooling (frameworks, autoscaling, CI/CD, model serving).
- Beating any benchmark — we want to reproduce, not innovate.

## 2. Model

GPT-2 124M architecture, identical to nanoGPT's `model.py`:

| Hyperparameter | Value |
|---|---|
| Layers (`n_layer`) | 12 |
| Heads (`n_head`) | 12 |
| Embedding dim (`n_embd`) | 768 |
| Context length (`block_size`) | 1024 |
| Vocab size | 50257 (GPT-2 BPE via tiktoken) |
| Parameter count | ~124M |

Standard pieces: token + position embeddings, 12× transformer block (pre-LayerNorm → MHA → residual → pre-LayerNorm → MLP-4x → residual), final LayerNorm, LM head with **weight tying** to the token embedding.

Attention uses `F.scaled_dot_product_attention` for built-in flash attention.

## 3. Training Recipe

Mirrors `build-nanogpt`, scaled for a single 5090:

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | `betas=(0.9, 0.95)`, `weight_decay=0.1` (decoupled, applied to 2D+ params only) |
| Max LR | 6e-4 | |
| Min LR | 6e-5 | 10% of max |
| Warmup steps | 715 | linear from 0 to max_lr |
| Schedule | cosine decay | after warmup |
| Total steps | ~19,073 | = 10B tokens / 524288 tokens per step |
| Tokens per step | 524,288 | = 2^19, matches GPT-2 paper's batch |
| Grad clip | 1.0 | global L2 norm |
| Precision | bfloat16 mixed | bf16 has full range, no loss-scaling needed |
| `torch.compile` | yes | |
| Seed | 1337 | fixed for reproducibility |

**Achieving 524K tokens/step on one 5090:** `micro_batch_size * seq_len * grad_accum_steps == 524288`. Concrete values are tuned empirically on the 5090; the skeleton asserts the product matches.

- Starting guess: `micro_batch_size=32`, `seq_len=1024`, `grad_accum_steps=16` (32 × 1024 × 16 = 524288).
- If OOM, halve `micro_batch_size` and double `grad_accum_steps`.

**Throughput target (rough, for sanity-checking):** 50-80k tokens/sec on a 5090 with bf16 + compile. At 524K tokens/step and ~70k tok/s, that's ~7.5s/step → 19,073 steps × 7.5s ≈ **40 hours** for the full run. Will measure for real on the 5090.

## 4. Repo Layout

Flat layout, close to nanoGPT — simpler for learning than `src/` package layout.

```
joey-minimal-gpt2/
├── README.md                    # Top-level overview, quickstart
├── pyproject.toml               # uv-managed deps
├── .python-version              # 3.12
├── .gitignore                   # ignores data/, logs/, checkpoints/, .venv/
│
├── docs/                        # Learning material
│   ├── 01-architecture.md       # GPT-2 block-by-block (attention, MLP, residual stream)
│   ├── 02-data-pipeline.md      # FineWeb-Edu, BPE, sharding format
│   ├── 03-training-recipe.md    # LR schedule, optimizer, grad accum, mixed precision
│   ├── 04-hardware-5090.md      # Memory math, throughput, batch tuning, what we skipped and why
│   └── 05-eval-and-sampling.md  # Val loss, HellaSwag, generation
│
├── model.py                     # GPT, Block, CausalSelfAttention, MLP — heavily commented
├── config.py                    # GPTConfig + TrainConfig dataclasses
├── data.py                      # Sharded data loader (DDP-ready in shape)
├── train.py                     # Main training loop (single-GPU)
├── eval_hellaswag.py            # Zero-shot HellaSwag scoring
├── sample.py                    # Load checkpoint and generate text
├── utils.py                     # LR schedule, device detect, CSV logger, seed
│
├── scripts/
│   ├── prep_fineweb_edu.py      # Download + tokenize → data/edu_fineweb10B/*.bin
│   └── prep_shakespeare.py      # Tiny dataset for Mac smoke tests
│
├── tests/
│   ├── test_model.py            # Forward pass shapes, param count, determinism
│   ├── test_data.py             # Shard loader correctness
│   └── test_smoke_train.py      # 20-step training on shakespeare; loss must drop
│
├── data/                        # gitignored — token .bin shards
├── logs/                        # gitignored — train.csv, samples
└── checkpoints/                 # gitignored — model_*.pt
```

## 5. Data Pipeline

### Source
HuggingFace dataset `HuggingFaceFW/fineweb-edu`, config `sample-10BT` (~10B tokens of educational web text).

### Tokenizer
`tiktoken` with the `gpt2` encoding (50257 BPE merges). EOT token id = 50256, used as a document separator.

### Sharding
`scripts/prep_fineweb_edu.py`:
1. Streams documents from HF in chunks.
2. Tokenizes in a `multiprocessing.Pool` (CPU-bound).
3. Concatenates tokens, separated by EOT.
4. Writes 100M-token shards as raw `uint16` `.bin` files under `data/edu_fineweb10B/`.

File naming: `edufineweb_{split}_{NNNNNN}.bin`. Shard 0 is reserved for validation.

Result: ~100 train shards + 1 val shard, ~20GB total on disk.

### Loader
`data.py` defines a `DataLoaderLite` (name from karpathy):

- Constructor: `(split, B, T, rank=0, world_size=1)`.
- Lists matching shards on disk, opens the first as a `np.memmap(dtype=np.uint16)`.
- `next_batch()` returns `(x, y)` tensors of shape `(B, T)` where `y` is `x` shifted by one position.
- Advances position by `B * T * world_size`; on shard exhaustion, advances to next shard.
- DDP-ready: rank/world_size make each process read a non-overlapping slice.

## 6. Training Loop

`train.py` is a flat script, not a class. Reading it top-to-bottom should show every step of a real training iteration.

Pseudocode:

```python
cfg = parse_train_config()
seed_everything(cfg.seed)
device = detect_device()                 # cuda > mps > cpu
model = GPT(GPTConfig()).to(device)
model = torch.compile(model)
optimizer = configure_adamw(model, ...)  # weight-decay only on 2D+ params
train_loader = DataLoaderLite("train", cfg.B, cfg.T)
val_loader   = DataLoaderLite("val",   cfg.B, cfg.T)
logger = CSVLogger("logs/train.csv")

for step in range(cfg.max_steps):
    # --- periodic eval ---
    if step % cfg.val_interval == 0:
        val_loss = run_val(model, val_loader, cfg.val_iters)
        logger.log_val(step, val_loss)
    if step % cfg.hella_interval == 0:
        acc = run_hellaswag(model, device)
        logger.log_hella(step, acc)
    if step % cfg.save_interval == 0:
        save_checkpoint(model, optimizer, step)

    # --- training step ---
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro in range(cfg.grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / cfg.grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = lr_schedule(step, cfg)
    for g in optimizer.param_groups: g["lr"] = lr
    optimizer.step()

    dt = time.time() - t0
    tps = cfg.B * cfg.T * cfg.grad_accum_steps / dt
    logger.log_train(step, loss_accum.item(), lr, dt*1000, tps, grad_norm.item())
```

## 7. Eval

### Validation loss
Every `val_interval` steps, evaluate `val_iters` batches on the held-out shard in `torch.no_grad()` + `model.eval()`. Mean cross-entropy goes to `train.csv` as `val_loss`.

### HellaSwag (zero-shot)
`eval_hellaswag.py`:
- Downloads HellaSwag val set (~10k examples).
- For each example: context + 4 candidate endings. Score each by per-token NLL of the ending given the context. Pick the lowest-NLL ending. Compare to gold label.
- Returns accuracy. Random baseline = 25%; GPT-2 124M target ≈ 29-30%.

Called from `train.py` every `hella_interval` steps, and runnable standalone for analysis.

### Sampling
`sample.py` loads a checkpoint and generates text. Supports `--temperature`, `--top_k`, `--max_tokens`, `--prompt`, `--n_samples`, `--seed`. Useful for eyeballing model quality over time.

## 8. Logging

Plain stdout + a single `logs/train.csv` file. Columns:

```
step, kind, loss, val_loss, hella_acc, lr, dt_ms, tokens_per_sec, grad_norm
```

`kind` is one of `train` / `val` / `hella` — only the relevant columns are filled per row. Easy to load later with `pandas.read_csv()` and plot. Zero external dependencies, zero accounts.

## 9. Testing Strategy

**Three tiers:**

1. **Unit-fast** (Mac CPU, seconds):
   - `test_model.py`: forward pass shapes, finite logits, param count == 124M ± rounding, deterministic generation with fixed seed.
   - `test_data.py`: loader contiguous (`y[i] == x[i+1]`), no overlap across calls, correct dtype, rank sharding splits cleanly.
2. **Smoke train** (Mac MPS or CPU, ~30s):
   - `test_smoke_train.py`: 20 steps on tiny Shakespeare, assert final loss < initial * 0.9. Catches broken plumbing before pushing to the 5090.
3. **Manual integration on 5090** (not automated, documented in `docs/04-hardware-5090.md`):
   - Step 1: run `prep_fineweb_edu.py`, verify shard count and total token count.
   - Step 2: run 100 training steps, verify tokens/sec in expected range.
   - Step 3: full run.

## 10. Error Handling Philosophy

Trust internal code, validate only at boundaries.

- CLI args validated at startup in dataclass `__post_init__` (e.g., assert `B * T * grad_accum_steps == total_batch_tokens`).
- Boundary file I/O wraps errors with actionable messages ("expected shards at `data/edu_fineweb10B/`, found nothing — did you run `scripts/prep_fineweb_edu.py`?").
- CUDA OOM: do not catch. Let it crash. The fix is to lower `micro_batch_size` and re-run — that's a human decision.
- Resume from checkpoint: explicit `--resume <path>` flag, not auto-discovery.
- HF downloads: no retry loops; re-run the script (HF cache is resumable).

## 11. Deliberately Out of Scope (with rationale)

This is intentionally documented for the learner — knowing what we *didn't* use matters.

### DDP (DistributedDataParallel)
Multi-GPU data parallelism. Each GPU holds a full model copy, processes a different slice of the batch, all-reduces gradients. `torchrun --nproc_per_node=N train.py`. We thread `rank`/`world_size` through the data loader so adding DDP later is ~5 extra lines, but the actual `DDP(model)` wrapper is not present.

### ZeRO / FSDP / pipeline / tensor parallelism
Strategies for models that don't fit on one GPU. ZeRO (DeepSpeed) and FSDP (PyTorch) shard parameters/gradients/optimizer states across GPUs. Pipeline parallelism puts different layers on different GPUs. Tensor parallelism splits individual layers. A 124M bf16 model is ~250MB — irrelevant at this scale.

### Training framework wrappers
- **DeepSpeed** (heavy, for huge models, includes ZeRO).
- **Accelerate** (HuggingFace, lighter — `accelerator.prepare()` abstracts device/distribution).
- **Lightning** (`LightningModule` with `training_step()`).
- **Fabric** (Lightning's lower-level alternative).

All hide the actual mechanics behind APIs. For learning we want to *see* `loss.backward()` and `optimizer.step()` written out in a flat script.

### `torch.compile` cache management
`torch.compile()` JIT-compiles model graphs via TorchInductor for 1.5-2× training speedups. First step is slow (compiling); afterwards it caches to `~/.cache/torch/inductor/`. We call `torch.compile()` and ignore the cache — fine for a single project.

### Checkpoint compression / safetensors / sharding
Plain `torch.save()` to `.pt`. Disk is cheap; we won't have thousands of checkpoints.

### A `Trainer` class abstraction
OOP-ifying the training loop hides the sequence of operations. The flat script is the point — you can read it top to bottom and see every step.

### LR finder
fast.ai trick to pick LR empirically by exponentially ramping it up. Not needed; GPT-2's known-good recipe specifies `max_lr=6e-4`.

### EMA (exponential moving average of weights)
Maintain a slow-moving weight copy for inference (smoother loss landscape, better generalization). Common in diffusion and SSL. Standard GPT-2 doesn't use it. One less moving part.

## 12. Hardware Notes (RTX 5090, 32GB)

- bf16 native support (Blackwell): use it, no loss scaling required.
- FP4/FP8 tensor cores exist but we won't use them — adds complexity for marginal benefit at 124M scale and we'd need careful numerics work.
- Memory budget at micro_batch=32, seq=1024:
  - Params (fp32 master + bf16) ≈ 750 MB
  - Optimizer states (AdamW m + v, fp32) ≈ 1 GB
  - Activations (12 layers × batch × seq × hidden × ~5 buffers × 2 bytes) ≈ 6-10 GB
  - Plenty of headroom in 32 GB.
- Expect 50-80k tokens/sec. Full 10B run ≈ 40 hours wall clock (rough estimate, will measure).

## 13. Dependencies

Managed by `uv` in `pyproject.toml`:

- `torch>=2.5` (for stable `torch.compile`, native bf16 SDPA)
- `tiktoken`
- `datasets` (HuggingFace, for FineWeb-Edu)
- `numpy`
- `requests` (HellaSwag download)
- `tqdm`

Dev:
- `pytest`
- `ruff` (lint + format)

Python: 3.12.

## 14. Implementation Order (high level)

1. Skeleton: `pyproject.toml`, `.gitignore`, `.python-version`, empty file structure.
2. `model.py` + `config.py` + `test_model.py`. Verify forward pass and param count on Mac.
3. `scripts/prep_shakespeare.py` + `data.py` + `test_data.py`. Smoke loader on Mac.
4. `utils.py` (LR schedule, device, CSV logger, seed).
5. `train.py` (single-GPU loop) + `test_smoke_train.py`. Run 20 steps on Shakespeare end-to-end.
6. `sample.py`. Generate from the smoke-test checkpoint.
7. `scripts/prep_fineweb_edu.py`. Documented but run on 5090, not Mac.
8. `eval_hellaswag.py`.
9. Five `docs/*.md` files.
10. `README.md` with quickstart for both Mac dev and 5090 training.

Detailed sequencing and review checkpoints will live in the implementation plan.
