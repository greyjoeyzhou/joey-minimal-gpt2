# Hardware: Training on a Single RTX 5090

This is the "how do I actually run it" companion to the training recipe.

## The 5090 spec sheet (relevant bits)

- 32 GB GDDR7 VRAM
- ~1.8 PFLOPS bf16/fp16 tensor cores
- ~3.4 PFLOPS fp4 (we don't use)
- Blackwell architecture: native bf16, full SDPA / flash attention support.

## Memory budget at 124M model

Rough numbers for `micro_batch=32, seq_len=1024`:

| Component | Size |
|---|---|
| Model params (bf16 + fp32 master copy) | ~750 MB |
| AdamW state (m, v in fp32) | ~1 GB |
| Activations (12 layers × B × T × n_embd × ~5 buffers × 2 bytes) | ~6-10 GB |
| KV / scratch buffers | ~1 GB |
| **Total** | **~10-12 GB** |

That leaves ~20 GB of headroom on 32 GB. You can comfortably push
`micro_batch_size` to 64 or even 128 before activations dominate.

## Tuning playbook

1. Run prep first: `uv run python scripts/prep_fineweb_edu.py`. Verify ~100
   shards in `data/edu_fineweb10B/`.
2. Run a smoke training pass — *very few steps*, default config:
   ```bash
   uv run python train.py --max_steps 50 --val_interval 100 --hella_interval 100 --save_interval 100
   ```
3. Watch `nvidia-smi` in another shell. Note GPU memory usage at steady state.
4. If memory is < 25 GB: double `micro_batch_size`, halve `grad_accum_steps`
   so `tokens_per_step` stays at 524288. Restart.
5. If you OOM: halve `micro_batch_size`, double `grad_accum_steps`. Restart.
6. Aim for the highest `micro_batch_size` that fits without OOM, leaving
   ~10% memory headroom (so `torch.compile`'s graph recompiles for varied
   shapes don't push you over).
7. Note the throughput (`tokens_per_sec` in train.csv). Reasonable target:
   50-80k tok/s. If you're at 20k, something is wrong.

## Throughput sanity checks

If `tokens_per_sec` is much lower than expected:

- `torch.compile` is disabled (check args, expect ~1.5x penalty without it).
- bf16 autocast is disabled (check `device=='cuda'` in train.py, expect ~2x).
- Data loader is the bottleneck (rare with mmap; would show up as low GPU
  utilization).
- Logging too frequently (we flush CSV every step, but that's tiny).

## Full run wall clock estimate

At 524288 tokens/step × 19073 steps = 10B tokens.

| tokens/sec | wall time |
|---|---|
| 50,000 | ~55 hours |
| 70,000 | ~40 hours |
| 100,000 | ~28 hours |

So expect a roughly 1.5-2 day run, end to end. Worth doing a 100-step
canary run first to project the actual rate.

## Resume after interruption

```bash
uv run python train.py --resume checkpoints/model_005000.pt
```

`save_interval=5000` (in TrainConfig) means we checkpoint every 5000 steps.
At the rates above that's every ~3-6 hours.

## What we deliberately *did not* set up (and why)

These appear in the design doc's "Out of Scope" list. Repeating here for
the operations perspective:

- **DDP** (multi-GPU on one box): not needed for 124M on one 5090. The
  loader and training loop are written rank-aware so adding it later is
  ~5 lines plus a `torchrun` launch.
- **FSDP / ZeRO / pipeline / tensor parallel**: for models that don't fit
  on one GPU. 124M fits 100x over. Not relevant here.
- **DeepSpeed / Accelerate / Lightning / Fabric**: framework wrappers. They
  hide what's happening in the training step behind APIs. We want to see
  every step — pedagogically vital, operationally simpler.
- **`torch.compile` cache management**: TorchInductor caches compiled graphs
  to `~/.cache/torch/inductor/`. Across PyTorch upgrades this can grow or
  go stale. For one project on one machine we ignore it; if it ever gets
  in the way, `rm -rf ~/.cache/torch/inductor/`.
- **Checkpoint compression / safetensors**: plain `.pt` is fine. ~500 MB per
  checkpoint, ~4 checkpoints across a run.
- **A `Trainer` class abstraction**: hides the loop. Don't want that here.
- **LR finder**: we use the GPT-2 paper's published LR, which is known to
  work.
- **EMA weights**: not used in standard GPT-2 training.

## When to graduate to a "real" setup

If you start training a 7B model, or do multi-machine training, or want
hyperparameter sweeps, you'll need to revisit the choices above. By that
point, the right move is usually to fork into a separate project (or use
Karpathy's `nanotron` / `litgpt` / etc.) rather than retrofit this one.

The goal here was "understand the mechanics," not "production training
platform." Once you understand the mechanics, the frameworks make sense.
