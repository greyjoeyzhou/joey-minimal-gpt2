# Attention Variants & Efficiency Mechanisms

Notes on what's changed in attention since GPT-2 — both *architectural variants* (MQA, GQA, MLA, SWA, hybrid SSM) that alter what the model computes, and *efficiency mechanisms* (Flash Attention, ring attention, activation checkpointing) that produce the same math faster.

Companion to [`attention.md`](./attention.md), which covers the baseline mechanism. This file covers what came after.

## The distinction that matters

Two ideas easily confused:

- **Modern variants** = architectural changes that alter what gets learned (different params, different attention patterns). Show up in the *model card*.
- **Efficient mechanisms** = compute optimizations that produce the *same math* faster or with less memory. Show up in the *kernel*.

A model can use both at once: LLaMA-3 uses GQA (variant) and Flash Attention (efficiency).

---

# Part 1: Modern variants

These all change the structure of what gets computed. They affect parameter count, what gets stored in the KV cache, and (sometimes) quality. The driving force for almost all of them is **KV cache memory at inference time** — not training compute.

## The KV cache memory problem

During autoregressive generation, you only run the model on the *new* token each step. To compute attention against all previous tokens, you cache their K and V projections — that's the **KV cache**. For a model with `n_layer × n_head × head_dim × 2` (×2 for K and V) values *per token*, the cache grows linearly with context length.

For LLaMA-3 70B (80 layers, 64 heads, head_dim=128, MHA): each token would need 80 × 64 × 128 × 2 = 1.3M values = 2.6 MB in bf16. At 100K context = 260 GB. **Doesn't fit on any GPU.**

This is *the* dominant pressure on attention architecture in 2023–2025. Every variant below is mostly a different answer to "how do we shrink the KV cache without hurting quality?"

## Summary table

| Variant | Year | Used by | What changes | KV cache size (vs MHA) | Quality impact |
|---|---|---|---|---|---|
| **MHA** (standard) | 2017 | GPT-2 (us), GPT-3, LLaMA-1 | N query heads, N K heads, N V heads | 1× baseline | baseline |
| **MQA** | 2019 | PaLM, Falcon | N query heads, **1 shared K head, 1 shared V head** | **1/N×** (e.g. 32× smaller) | small drop |
| **GQA** | 2023 | LLaMA-2 70B+, LLaMA-3, Mistral, Mixtral, Qwen-2, Gemma-2 | N query heads, **G shared K/V groups** (G < N) | **G/N×** (typical 1/4 to 1/8) | near-MHA |
| **MLA** | 2024 | DeepSeek-V2, V3 | K/V compressed into low-rank latent, decompressed per head | **even smaller than GQA** | matches MHA |
| **SWA** | 2020 (Longformer), 2023 (Mistral) | Mistral 7B, Longformer | Each token attends to last W tokens only | bounded by W (constant!) | depends on use |
| **Hybrid SSM-Attention** | 2024 | Jamba, Hymba, Zamba | Mix of attention + Mamba layers | only attention layers have KV cache | varies |

## MQA — Multi-Query Attention (Shazeer, 2019)

**All query heads share a single K head and a single V head.**

```
Standard MHA:                       MQA:
  Q: (B, n_head, T, head_dim)         Q: (B, n_head, T, head_dim)
  K: (B, n_head, T, head_dim)         K: (B, 1,      T, head_dim)   ← shared
  V: (B, n_head, T, head_dim)         V: (B, 1,      T, head_dim)   ← shared
```

When computing attention, the single K/V gets broadcast across all query heads. The math works identically.

**Why it helps**: KV cache is N× smaller (where N = n_head). For a 32-head model, 32× reduction.

**Why it costs quality**: K and V can no longer specialize per head. Every head queries the same shared "memory." The diversity-of-specialization benefit of multi-head is partially lost.

**In practice**: PaLM (Google, 2022) showed MQA gives big inference speedups for modest quality loss. Falcon (TII) used it. But the quality loss was noticeable enough that GQA replaced it.

## GQA — Grouped-Query Attention (Ainslie et al., 2023)

**The compromise that won.** N query heads, G groups of K/V heads, where 1 < G < N.

```
Example: n_head=32, G=8 groups → 32 query heads, 8 K heads, 8 V heads.
Group size = 32/8 = 4 → 4 query heads share each K/V pair.

  Q: (B, 32, T, head_dim)
  K: (B,  8, T, head_dim)   ← 4 query heads share each K
  V: (B,  8, T, head_dim)   ← 4 query heads share each V
```

**Why it's everywhere**: Recovers most of MHA's quality at most of MQA's speed. **GQA is now the default for essentially every modern open model.**

Concrete: LLaMA-3 70B has 64 query heads, 8 K/V heads → 8× KV cache reduction with negligible quality cost.

## MLA — Multi-head Latent Attention (DeepSeek-V2/V3, 2024)

More sophisticated approach: instead of *sharing* K/V across heads, **compress K and V into a low-rank latent representation**.

The idea, simplified:
1. Project the input into a small latent vector (much smaller than `n_embd`).
2. Cache *the latent*, not the full K/V.
3. When computing attention, decompress the latent back into per-head K and V via learned matrices.

Result: KV cache stores just the latent (very small), but you still get per-head specialization at compute time.

**Why it's notable**: DeepSeek-V3 has 671B total params but only 37B active per token (also MoE). MLA's KV cache savings are part of what makes long-context inference economically viable.

**Why it hasn't replaced GQA yet**: complex to implement, more recent. Likely the next default — the quality/efficiency tradeoff is reportedly better than GQA.

## SWA — Sliding Window Attention (Mistral, 2023; Longformer, 2020)

**Each token attends to only the last W tokens, not all previous tokens.**

```
Causal mask becomes a band, not a triangle:

  Standard causal:           Sliding window (W=3):
    [1, 0, 0, 0, 0]            [1, 0, 0, 0, 0]
    [1, 1, 0, 0, 0]            [1, 1, 0, 0, 0]
    [1, 1, 1, 0, 0]            [1, 1, 1, 0, 0]
    [1, 1, 1, 1, 0]            [0, 1, 1, 1, 0]    ← can't see token 0
    [1, 1, 1, 1, 1]            [0, 0, 1, 1, 1]
```

**Why it helps**: Attention becomes `O(T × W)` instead of `O(T²)`. KV cache bounded by W (constant!) instead of growing with context.

**Why it costs**: Tokens beyond W get cut off entirely. *But* — through stacking layers, information can propagate further than W. After 2 layers with W=4096, effective receptive field is 8192. After 8 layers, 32K. The model still has "global" information, just propagated through layer stacking.

**Mistral 7B's recipe**: SWA with W=4096, combined with full attention every few layers.

**Trend**: Pure SWA has had mixed results vs full attention. Modern models often use SWA in *some* layers and full attention in others, getting the best of both.

## Hybrid SSM-Attention (Jamba, Hymba, 2024)

**Mix transformer layers with State-Space Model (Mamba) layers.**

State-Space Models are an alternative to attention that operate in `O(T)` time with constant memory (no KV cache at all). Their downside: weaker at "recall" tasks (looking up a specific past token). Attention's strength.

Hybrid models like **Jamba** (AI21, 2024): mostly Mamba layers (cheap, fast), sprinkled with attention layers (expensive but powerful at recall). Long-context efficiency from Mamba, retrieval ability from attention.

**Status**: actively researched, not yet dominant. Worth watching.

## Differential Attention (Microsoft, 2024)

Split each attention head into two, subtract the softmax outputs:

```
diff_attn = softmax(Q_1 K_1ᵀ / √d) - λ · softmax(Q_2 K_2ᵀ / √d)
```

Intuition: standard attention assigns non-zero weight to irrelevant tokens ("attention noise"). The second softmax learns to model that noise, and subtracting it leaves cleaner attention. Reported quality improvements on long-context recall. Whether this generalizes is unclear.

## Cross-cutting: positional encoding lives inside attention

Modern variants almost universally pair with **RoPE** (rotary position embeddings) instead of learned absolute positions like our `wpe`. RoPE happens *inside* attention — it rotates Q and K by position-dependent angles before the matmul. See [`embedding.md`](./embedding.md) §7.

---

# Part 2: Efficient mechanisms (same math, faster)

These produce identical (or near-identical) outputs to standard attention but compute it more efficiently. Transparent — you don't change the model definition, you change the kernel that runs.

## The memory bottleneck of standard attention

The naive implementation in [`attention.md`](./attention.md) §3 materializes an `(B, n_head, T, T)` tensor for the attention scores. For:
- B = 32 (our micro batch)
- n_head = 12
- T = 1024
- fp32 bytes per element = 4

That's `32 × 12 × 1024 × 1024 × 4 = 1.6 GB` per layer, just for the attention matrix. Times 12 layers = 19 GB. **Just for the intermediate.**

Fine at T=1024 with a small model. A disaster at T=8192 with a 70B model — the attention matrix alone would exceed any single GPU's memory.

The problem isn't the *math*, it's that PyTorch's naive kernel reads/writes the full attention matrix to HBM (the GPU's main memory). HBM is ~1.5 TB/s on a 5090. Compute is fast enough; **memory traffic is the bottleneck.**

## Flash Attention v1 (Dao et al., 2022)

**The key paper.** Reorders attention to:

1. **Never materialize the full attention matrix in HBM.**
2. Compute attention in *tiles* that fit in SRAM (on-chip cache, much smaller but much faster than HBM).
3. Use **online softmax** — a streaming softmax algorithm that computes the final result correctly even though each tile only sees a slice of the row.
4. Recompute attention in the backward pass (instead of storing it).

```
Standard attention memory pattern:
  read Q, K from HBM            }
  compute Q@Kᵀ                  } each step reads/writes
  write scores to HBM           } a huge (T, T) matrix
  read scores from HBM          } to/from slow HBM
  softmax                       }
  write attn to HBM             }
  read attn from HBM            }
  compute attn @ V              }
  write output                  }

Flash Attention memory pattern:
  for each block of Q rows:
    load Q block into SRAM
    for each block of K, V rows:
      load K, V blocks into SRAM
      compute partial Q@Kᵀ in SRAM
      update running softmax statistics
      compute partial output
    write final output for this Q block to HBM
  → full attention matrix never exists outside SRAM
```

**Result**: 2-4× speedup on standard attention, **dramatically** less memory. Enables training with much longer sequences on the same hardware.

**The catch**: requires hand-written CUDA. PyTorch can't auto-fuse this. That's why `F.scaled_dot_product_attention` dispatches to a *specific* implementation based on your GPU.

## Flash Attention v2 (Dao, 2023)

Refines v1:
- **Better parallelization across the T dimension.** v1 parallelized across batch and head; v2 also parallelizes within a sequence, which helps when batch is small but T is large (modern long-context training).
- Reduces non-matmul ops (which on H100s are the bottleneck because matmuls are so fast).
- Better support for variable-length sequences (important for packed batches).

Another **~2× speedup** over v1.

## Flash Attention v3 (Dao et al., 2024)

Hopper-specific (H100/H200/B200) optimizations:
- Uses **asynchronous memory transfers** (`cp.async`) to overlap memory loads with compute.
- Uses **WGMMA** (Warpgroup Matrix Multiply Accumulate) — H100's new async tensor core ops.
- FP8 support for even more throughput.

Pushes Flash Attention to ~75% of theoretical peak on H100. **For an attention layer to hit 75% MFU is nearly miraculous** — it used to be 10-20%.

## How Flash Attention plumbs into our code

`F.scaled_dot_product_attention` (`model.py:81`) automatically picks the best implementation based on:
- GPU architecture (Ampere/Hopper/Blackwell)
- Sequence length, dtype, causal flag
- Whether dropout is in use

On the 5090 (Blackwell) with bf16, `is_causal=True`, our shapes: it dispatches to **Flash Attention v2 or v3**. We get all the optimization for free, no kernel writing.

Force a specific backend for benchmarking or debugging:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force Flash Attention v2
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## Other efficiency mechanisms

### Memory-Efficient Attention (xformers)

Predates Flash Attention v1 by ~year. Same core idea (tile-and-stream), different implementation. Mostly superseded by Flash Attention. Still available as a dispatch option in PyTorch.

### Linear / sub-quadratic attention (approximations)

Change the math — approximations, not exact. Worth knowing but **not currently winning**:

- **Performer** (2020): approximates softmax with random feature maps. `O(T)` instead of `O(T²)`.
- **Linear Transformer**: drops the softmax entirely, uses kernel feature maps. `O(T)`.
- **Reformer**: locality-sensitive hashing to attend to similar tokens only.

Looked promising in 2020-2021 but **mostly lost to Flash Attention**. Flash made exact softmax attention cheap enough that approximations aren't worth the quality hit. Might come back for very long contexts (1M+ tokens) where even Flash Attention struggles.

### Sparse attention patterns

Specify in advance *which* positions can attend to which. Not approximations of softmax, but restrictions on the attention pattern:

- **Longformer** (2020): sliding window + a few global tokens.
- **BigBird** (2020): sliding window + random + global.
- **Sparse Transformer** (OpenAI, 2019): strided patterns.

Now mostly used for specific long-document tasks. General LMs prefer SWA-style (which is a kind of sparse pattern but with clean GPU-friendly structure).

### Ring Attention (Liu et al., 2023)

**For training at extreme context lengths** (1M+ tokens). Splits the KV cache across multiple GPUs in a ring topology — each GPU computes attention against its local chunk, then passes K/V to the next GPU around the ring. Combined with sequence parallelism.

This is how Gemini 1.5 (2M tokens) and some recent long-context training is done. Not relevant for our 1024-context model.

### Activation checkpointing (a.k.a. gradient checkpointing)

**Trade compute for memory.** Instead of storing all forward activations for backward, store only some, recompute the rest. Attention's `(T, T)` activation is the main target — even with Flash Attention, the backward pass needs *some* memory.

In PyTorch: `torch.utils.checkpoint.checkpoint(fn, x)`. Roughly 30% slower backward, 30-50% less peak memory. Standard practice for training large models. Not used here because our model is small enough not to need it.

### KV cache itself (inference-time, worth noting)

Inference optimization: during generation, every token only needs to compute its own Q against all previous K/V. Cache the K/V from previous tokens once, only compute new K/V/Q each step. Turns generation from O(T²) per token into O(T).

`model.py:282-323`'s `generate()` does **not** implement KV cache — it just re-runs the full forward pass each step. Generation is wasteful, code is simpler. For real production inference (vLLM, TensorRT-LLM, llama.cpp), KV cache is essential.

---

## What we do, what we don't

| Mechanism | Status | Notes |
|---|---|---|
| Standard MHA | ✓ Used | We follow GPT-2 faithfully |
| Flash Attention | ✓ Used | Automatically via `F.scaled_dot_product_attention` |
| MQA / GQA / MLA | ✗ Not used | Modern variants, not GPT-2's design |
| SWA | ✗ Not used | Irrelevant at T=1024 |
| KV cache (inference) | ✗ Not implemented | `generate()` is naive |
| Activation checkpointing | ✗ Not used | Model too small to need it |
| Ring attention | ✗ Not used | Single-GPU, short context |
| RoPE (inside attention) | ✗ Not used | We use learned `wpe` |

## If you wanted to modernize

Highest-impact changes in roughly this order:

1. **Add KV cache to `generate()`** — biggest practical inference speedup, no quality change. Maybe 50 lines of code.
2. **Switch to RoPE** — covered in [`embedding.md`](./embedding.md). Unlocks context-length extension at inference.
3. **Switch to GQA** — would barely matter at our 12-head scale, but at 32+ heads it's huge for KV cache.
4. **Add SWA** — irrelevant at T=1024; relevant if you push context to 16K+.
5. **Try MLA** — most aggressive KV cache reduction, complex to implement.

## The two-axis mental model

When you read about a new attention paper, ask:

1. **Does it change the math?** → architectural variant. Affects what the model learns. Goes in the model card.
2. **Does it preserve the math, just compute it cheaper?** → efficiency mechanism. Transparent. Goes in the kernel.

Some papers do both (rare). Most clearly fall into one bucket. Once you see this, the whole landscape becomes easier to navigate.
