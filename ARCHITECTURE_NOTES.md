# Architecture Notes: GPT-2 → Modern LLM

A learning reference from training v0 through v3 on FineWeb-Edu (10B tokens).

---

## The four models at a glance

| | v0 (GPT-2) | v1 (modern) | v2 (nanowhale) | v3 (v1 + MoE) |
|---|---|---|---|---|
| **File** | `model.py` | `model_v1.py` | `model_v2.py` | `model_v3.py` |
| **Params (total)** | ~124M | ~114M | ~106M | ~155M |
| **Active/token** | ~124M | ~114M | ~60M | ~116M |
| **Layers** | 12 | 12 | 8 | 12 |
| **Norm** | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| **Position** | Learned abs. | RoPE | RoPE + NoPE | RoPE |
| **Attention** | MHA | GQA | MLA | GQA |
| **FFN** | GELU dense | SwiGLU dense | SwiGLU MoE | SwiGLU MoE |
| **Residual** | Standard | Standard | Hyper-Connections | Standard |
| **Context** | 1024 | 2048 | 2048 | 2048 |
| **HellaSwag** | 30.6% | 30.2% | — | TBD |

---

## Component by component

### 1. RMSNorm (v1+)

**What:** Normalization without mean-centering.

```
LayerNorm(x) = (x − mean) / sqrt(var + ε) × weight + bias
RMSNorm(x)  =  x          / sqrt(mean(x²) + ε) × weight
```

**Why it works:** The stability benefit of LayerNorm comes from the RMS
scaling, not the mean-centering. Dropping mean-centering saves ~15% of the
normalization compute with negligible quality loss.

**Used by:** Llama 2/3, Mistral, Gemma, all post-2022 open models.

---

### 2. RoPE — Rotary Position Embeddings (v1+)

**What:** Encodes position as a rotation in Q and K, instead of a learned
lookup table (wpe).

For position m and frequency index i:
```
angle(m, i) = m / (theta ^ (2i / head_dim))
Q_rotated[m] = Q[m] × e^(j × angle)
```

**Key property:** After rotation, the dot product `Q_m · K_n` depends only
on the *relative* position `(m - n)`, not the absolute positions. The model
naturally learns relative positional patterns.

**Why better than learned wpe:**
- No parameter table to train.
- Generalizes to lengths beyond training context (with tricks like YaRN).
- v0's learned wpe breaks completely at lengths > `block_size`.

**RoPE + NoPE split (v2):** Each head dimension is split into rope dims
(position-aware) and nope dims (content-only, no rotation). Gives the model
the option to use or ignore position information per dimension.

**Used by:** Llama 2/3, Mistral, Gemma, Falcon, GPT-NeoX.

---

### 3. SwiGLU (v1+)

**What:** A gated feedforward network replacing the standard GELU MLP.

```
# v0 dense MLP (2 projections):
out = W2(GELU(W1(x)))

# SwiGLU (3 projections):
out = W_down( SiLU(W_gate(x)) ⊗ W_up(x) )
```

**Why:** The gate branch `SiLU(W_gate(x))` modulates the up branch
element-wise — each hidden unit's contribution is conditional on the input.
More expressive per parameter than unconditional GELU.

**Hidden dim scaling:** Uses `2/3 × 4 × n_embd` (not `4×`) to keep total
parameter count comparable to a 2-projection MLP:
- v0 MLP:  `2 × (768 × 3072)` = 4.72M params per layer
- SwiGLU:  `3 × (768 × 2048)` = 4.72M params per layer  ← same

**SiLU:** `silu(x) = x × sigmoid(x)`. Smooth, non-monotonic. Also called "swish".

**Used by:** Llama 2/3, Mistral, PaLM, Gemma.

---

### 4. GQA — Grouped-Query Attention (v1+)

**What:** Fewer key/value heads than query heads. Each KV head is shared
by `n_head / n_kv_head` query heads.

```
MHA:  n_kv_head = n_head = 12      (no sharing)
GQA:  n_kv_head = 4, n_head = 12  (3 Q heads per KV group)
MQA:  n_kv_head = 1               (extreme — all Q heads share 1 KV)
```

**Why:** The KV cache is the dominant memory cost at inference. Every
autoregressive decoding step must store past K and V for every layer:
```
KV cache per token = n_kv_head × head_dim × 2 × n_layer × 2 bytes
MHA (n_kv_head=12): 12 × 64 × 2 × 12 × 2 = 36,864 bytes per token
GQA (n_kv_head=4):   4 × 64 × 2 × 12 × 2 = 12,288 bytes per token  ← 3× smaller
```

**Quality:** Essentially unchanged vs MHA in practice. The saving is entirely
at inference, not training.

**Used by:** Llama 3, Mistral, Falcon 180B.

---

### 5. MLA — Multi-head Latent Attention (v2)

**What:** Compresses K and V through a shared low-rank bottleneck before
caching, instead of caching full K and V.

```
# Standard GQA: cache K, V directly
# MLA: project x → c_KV (kv_lora_rank=96 dims), cache that + k_rope
c_KV = W_down(x)       # (B, T, 96)  ← what gets cached
k_content = W_up_k(c_KV)
v         = W_up_v(c_KV)
k_rope    = W_kr(x)    # (B, T, 32)  ← also cached (positional part)
```

**KV cache comparison:**
```
MHA  (n_head=8):  2 × 8 × 96 = 1536 dims per token
GQA  (n_kv=1):   2 × 1 × 96 =  192 dims per token
MLA:              96 + 32    =  128 dims per token  ← ~12× smaller than MHA
```

**NoPE dims:** Head dimensions that skip RoPE rotation. Lets the model use
content-only features that are not position-dependent.

**Primary benefit:** Inference memory, not training quality.

**Used by:** DeepSeek-V2/V3, nanowhale.

---

### 6. MoE — Mixture of Experts (v2, v3)

**What:** Replaces the dense FFN with a bank of expert networks. Each token
routes to a subset of experts via a learned router.

```
# Dense FFN: all params active for every token
out = FFN(x)

# MoE FFN: only k+1 of N+1 experts active per token
scores = softmax(router(x))              # (B, T, n_routed)
selected = top_k(scores)
out = shared_expert(x) + Σ weight_i × expert_i(x)  # for selected i only
```

**Why this matters — the parameter/compute tradeoff:**

This is the core MoE insight. At the same *active compute* per token, MoE
provides more total *parameter capacity*:

```
Dense v1:   114M total = 114M active per token
MoE v3:     155M total = 116M active per token  (+39M "free" capacity)
```

The 39M non-active params (routed experts that didn't fire) cost nothing at
inference but let different experts specialize:
- Expert 0 → punctuation / syntax
- Expert 1 → factual recall
- Expert 2 → code / structure
- Expert 3 → rare vocabulary

(This specialization is empirically observed in Mixtral-style models.)

#### The compute-matching rule

**Never compare MoE total params to dense params.** Always match active compute:

```
(n_shared + n_experts_per_tok) × expert_size = dense_ffn_size
(1 + 2)                        × expert_size = 4.72M
                                 expert_size = 1.57M  →  intermediate ≈ 704
```

| Design | Total params | Active/token | Fair comparison? |
|---|---|---|---|
| v1 dense | ~114M | ~114M | reference |
| MoE param-matched | ~114M | ~68M | no — less compute per token |
| MoE compute-matched | ~155M | ~116M | yes — same FLOPs, more capacity |

#### Load balance loss

Without regularization, routers collapse: 1-2 experts handle everything,
the rest go unused. The auxiliary loss prevents this:

```
L_aux = router_scale × n_experts × Σ_i (f_i × P_i)

f_i = fraction of tokens dispatched to expert i  (discrete, not differentiable)
P_i = mean router probability for expert i        (differentiable)
```

`f × P` is the differentiable surrogate: when expert i is overloaded (high f_i),
reducing P_i lowers the loss, pushing the router away from it.

**Used by:** Mixtral, DeepSeek-V2/V3, Switch Transformer, GPT-4 (rumoured).

---

### 7. Hyper-Connections (v2)

**What:** Replaces the single residual stream `x += f(x)` with multiple
parallel streams. Each layer reads a learned weighted combination of all
streams as input, and distributes its output back to all streams.

```
# Standard residual:
x = x + f(norm(x))

# Hyper-connections (hc_expansion=2 streams):
# h: (B, T, 2, n_embd)
x_in = Σ softmax(alpha_i) × h[:,:,i,:]   # weighted input
out  = f(norm(x_in))
h[:,:,i,:] += sigmoid(beta_i) × out      # per-stream output update
```

**Initialized** so alpha[0] = high, others low → starts as standard residual.
The extra streams are learned skip connections that can bypass layers.

**Status:** Relatively new (2024), appears in nanowhale. Less proven than the
other components at scale. Adds `hc_expansion×` memory overhead for hidden states.

---

### 8. MTP — Multi-Token Prediction (v2)

**What:** An auxiliary head that predicts token `t+2` alongside the main
`t+1` prediction.

```
main_loss = CE(lm_head(x),   targets)        # predict t+1
mtp_loss  = CE(mtp_head(x),  targets[:,1:])  # predict t+2

loss = main_loss + 0.1 × mtp_loss
```

**Why:** Forces the model to represent multiple future tokens in its hidden
states, not just the immediate next token. Improves representations and
enables speculative decoding at inference.

**Used by:** DeepSeek-V3, Meta's MTP paper (2024).

---

## HellaSwag results

HellaSwag is a 4-way commonsense completion benchmark. Random = 25%.

| Model | HellaSwag | Notes |
|---|---|---|
| Random | 25.0% | baseline |
| GPT-2 small (124M, OpenAI) | ~29.4% | published result |
| **v0** (step 20,999) | **30.6%** | 10B tokens, GPT-2 arch |
| **v1** (step 19,072) | **30.2%** | 10B tokens, modern arch |
| **v3** | TBD | same FLOPs as v1, +39M capacity |

**Why v1 ≈ v0 despite better architecture?**
v1 has ~10M fewer params than v0 due to GQA reducing KV projections
(114M vs 124M). The architectural improvements offset but don't exceed
this param reduction at 10B tokens. The real v1 wins show up at inference
(smaller KV cache, longer context) and at larger token budgets.

---

## Training costs (RTX 5090, 32 GB)

### VRAM (micro_batch=16, seq_len=2048, one micro-step)

| Component | v1 | v3 |
|---|---|---|
| Weights (bf16) | 228 MB | 310 MB |
| Optimizer (fp32 ×3) | 1.37 GB | 1.86 GB |
| Gradients (bf16) | 228 MB | 310 MB |
| Activations (hidden + MLP) | 3.8 GB | 3.9 GB |
| **Total** | **~5.5 GB** | **~6.3 GB** |
| 5090 headroom | 26.5 GB free | 25.7 GB free |

Both fit with massive headroom. Activations are nearly identical because
active compute per token is matched.

### Training time (19,073 steps)

| | v1 | v3 (naive loop) |
|---|---|---|
| MLP kernels per layer | 3 (one big matmul) | 8 (small per-expert) |
| Step time (est.) | ~1s | ~2–3s |
| Full run (est.) | ~5–6 hours | ~10–16 hours |

**Why v3 is slower:** The Python expert dispatch loop launches many small
CUDA kernels on token subsets (~1/5 of batch each), instead of one large
matrix multiply. Small kernels underutilize tensor cores.

**Production fix — batched expert dispatch:**
```
# Naive (ours):  loop → N small matmuls
# Optimized:    sort tokens by expert → one large batched GEMM → scatter back
```
Frameworks like Megablocks and DeepSpeed MoE implement this. Reduces v3
overhead to ~1.2–1.5× v1 instead of 2–3×.

---

## Key papers

| Topic | Paper |
|---|---|
| RMSNorm | Zhang & Sennrich, 2019 — https://arxiv.org/abs/1910.07467 |
| RoPE | Su et al., 2021 — https://arxiv.org/abs/2104.09864 |
| SwiGLU | Shazeer, 2020 — https://arxiv.org/abs/2002.05202 |
| GQA | Ainslie et al., 2023 — https://arxiv.org/abs/2305.13245 |
| MLA | DeepSeek-V2, 2024 — https://arxiv.org/abs/2405.04434 |
| MoE (Switch) | Fedus et al., 2022 — https://arxiv.org/abs/2101.03961 |
| MoE scaling laws | Clark et al., 2022 — https://arxiv.org/abs/2202.01169 |
| Hyper-Connections | Zhu et al., 2024 — https://arxiv.org/abs/2409.19606 |
| MTP | DeepSeek-V3, 2024 — https://arxiv.org/abs/2412.19437 |
| Chinchilla (scaling) | Hoffmann et al., 2022 — https://arxiv.org/abs/2203.15556 |
| nanowhale | HuggingFaceTB — https://huggingface.co/HuggingFaceTB/nanowhale-100m-base |
