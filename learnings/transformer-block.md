# The Transformer Block — Residual Stream, LayerNorm, MLP

Notes on the three pieces of a transformer block that aren't attention: the **residual stream** (the design pattern that makes every block additive), **LayerNorm** (normalization applied before each sublayer), and the **MLP** (per-position feedforward that does most of the model's heavy lifting in parameters).

See also: [`attention.md`](./attention.md) (the other sublayer), [`abbreviations.md`](./abbreviations.md) (for what `ln_1`, `c_fc`, etc. stand for).

## 1. The block, as a whole

A transformer block in our model (`model.py:113-134`):

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

Two sublayers (attention and MLP), each preceded by a LayerNorm, each adding back into `x`. The whole model stacks 12 of these.

The crucial structural detail is the **pattern of each line**:

```
x = x + sublayer(norm(x))
```

Three things happen, in this order:
1. **Normalize** the current state with LayerNorm
2. **Compute** the sublayer (attention or MLP) on the normalized state
3. **Add** the result back into `x`

`x` itself is never replaced — it accumulates contributions. This is the residual stream pattern, and it's the most important architectural idea in modern deep networks (beyond what's specific to transformers).

## 2. The residual stream

### The big picture

Every block in the model reads from and writes to a shared `(B, T, C)`-shaped tensor that flows through the entire network. We call this the **residual stream**. It starts as the sum of token + position embeddings, gets *added to* by every attention and MLP sublayer, and ends up as the input to the final LayerNorm and language modeling head.

```
embeddings ──► block 1 ──► block 2 ──► ... ──► block 12 ──► ln_f ──► lm_head ──► logits
              │            │                     │
              │  attn      │  attn               │  attn
              │  +mlp      │  +mlp               │  +mlp
              └────────────┴─────────────────────┘
              all adding into the same C=768-dim stream
```

This view comes from **mechanistic interpretability** (Anthropic, Conjecture, others). The metaphor: think of the residual stream as a **bus** with 768 wires. Every block can read any subset of the wires (via `ln_1`/`ln_2` + sublayer projections), do some computation, and write back to any subset of the wires (via the sublayer's output projection). Information accumulates over depth.

Three consequences worth internalizing:

**(a) Every layer sees the input.** Because `x` carries forward unchanged through residual additions, even layer 12 has direct access to the token embeddings. A path from input to output that skips all 12 sublayers literally exists. This is what makes very deep networks trainable.

**(b) Layers compose by addition, not replacement.** A sublayer's contribution doesn't overwrite what came before — it adds. Multiple sublayers can independently contribute different signals to the same dimensions of `x`, and they sum.

**(c) The stream stays in the same dimensional space.** Every read and write uses `C=768`-dim vectors. This is why attention and MLP outputs are projected back to `n_embd` at the end (`c_proj`) — they have to fit back into the stream.

### Why residual connections matter (the ResNet origin)

Residual connections come from Kaiming He's **ResNet** paper (2015), which solved the "deep networks don't train" problem in computer vision. The observation:

If you stack 50 plain conv layers, the network trains *worse* than 20 layers. Not just diminishing returns — actively worse. Why? Gradients vanish through long chains of multiplications. The deepest layers can't learn.

He's fix: add a "skip" path that lets gradient (and forward signal) bypass each layer:

```
plain:    y = f(x)            ← layer can only transform
residual: y = x + f(x)        ← layer adds; x flows through unchanged
```

The transformer paper (2017) adopted this immediately. Without residual connections, you can't train transformers past ~3-4 layers. With them, you can train hundreds.

### Pre-LN vs post-LN — what changed from the original transformer

The original 2017 paper used **post-LN**:
```
x = LayerNorm(x + sublayer(x))
```
(normalize *after* the residual add)

GPT-2 onward use **pre-LN**:
```
x = x + sublayer(LayerNorm(x))
```
(normalize *before* the sublayer; the residual addition keeps `x` un-normalized)

This looks like a tiny change. It has enormous practical consequences:

| | Post-LN (original) | Pre-LN (modern) |
|---|---|---|
| Trains stably without warmup? | No — needs careful LR warmup or training diverges | Yes |
| Easy to scale to 12+ layers? | Hard | Easy |
| Identity path through the network? | No — every layer's output gets normalized | Yes — `x` flows through unchanged |
| Final LayerNorm at the end? | No (already normalized everywhere) | Yes (`ln_f`) — because the stream itself isn't normalized |

The reason pre-LN trains better: **each sublayer's input is well-behaved** (LayerNorm output has unit variance), and the residual addition doesn't get re-normalized, so gradients flow cleanly through the additive path. Post-LN puts the normalization *on the residual path*, which means gradients get squashed at every layer.

Almost every modern transformer uses pre-LN. The 124M GPT-2 we're reproducing uses pre-LN. The original 2017 paper's post-LN design is now considered a historical artifact.

### Why we need `ln_f` at the end

Because the stream is never normalized along the way (pre-LN normalizes only the *inputs to sublayers*, not the stream itself), the final state of `x` after 12 blocks has unbounded variance — it's the sum of 12 sublayer outputs plus the original embeddings. Before we hand it to `lm_head` to produce logits, we apply one final LayerNorm (`ln_f`, `model.py:163, 231`) to bring it to a well-behaved scale.

If you removed `ln_f`, training would still work but logits would have huge variance, softmax would saturate, gradients would vanish. `ln_f` exists *because* of the pre-LN design choice.

## 3. LayerNorm

### What it computes

For each token's `C`-dim vector independently:

```
Given input x ∈ ℝ^C:
  μ = mean(x)                ← scalar, mean across the C dimensions
  σ² = variance(x)           ← scalar, variance across the C dimensions
  x̂ = (x - μ) / √(σ² + ε)   ← normalized: mean 0, variance 1
  y = γ ⊙ x̂ + β              ← learned scale γ and shift β (both ∈ ℝ^C)
```

`γ` (called `weight` in PyTorch, the "gain") and `β` (called `bias`) are **learned per-feature** — they're vectors of length `C=768`, one of the few learned parameters in LayerNorm. They let the model un-do or rescale the normalization on a per-feature basis if doing so helps.

`ε` is a small constant (default `1e-5` in PyTorch) to prevent division by zero when the variance is tiny. Numerical hygiene, not a tunable hyperparameter.

### Per-token, NOT per-batch

This is the crucial distinction from **BatchNorm**:

- **BatchNorm** computes `μ` and `σ²` across the batch dim. Each feature's stats come from all examples in the batch.
- **LayerNorm** computes `μ` and `σ²` across the feature dim (`C`). Each example's stats come from just its own features.

Why LayerNorm for transformers?

1. **Independence of batch size.** Same forward pass at batch=1 as at batch=32. BatchNorm changes behavior with batch size, which is bad for sequence models where batches are awkwardly shaped.
2. **Independence of sequence length.** Each token's normalization doesn't depend on other tokens in the sequence. This matters because attention already does the cross-token mixing — normalization shouldn't add more dependencies.
3. **No train/eval mode difference.** BatchNorm tracks running statistics during training, uses them at eval — a constant source of bugs. LayerNorm is identical in train and eval.

### Where LayerNorm sits in our model

Three places (`model.py:122-125, 163`):

```python
self.ln_1 = nn.LayerNorm(config.n_embd)   # before attention, in each block
self.ln_2 = nn.LayerNorm(config.n_embd)   # before MLP, in each block
self.ln_f = nn.LayerNorm(config.n_embd)   # once at the end of the stack
```

So a 12-layer model has 12 × 2 + 1 = **25 LayerNorm instances**. Each has `2 × 768 = 1,536` parameters (γ and β). Total LN params: 25 × 1536 ≈ 38K — a rounding error compared to the 124M total. **LayerNorm parameters are cheap; the operation is cheap; the benefit is enormous.**

### Modern alternative: RMSNorm

LLaMA, Mistral, and most post-2023 models use **RMSNorm** (Zhang & Sennrich, 2019) instead of LayerNorm:

```
Standard LayerNorm:           RMSNorm:
  μ = mean(x)                   (no mean subtraction)
  σ² = variance(x)              rms² = mean(x²)
  y = γ⊙(x - μ)/√(σ²+ε) + β    y = γ ⊙ x / √(rms² + ε)
```

Differences:
- **No mean subtraction** — only normalizes by RMS.
- **No `β` (bias) parameter** — only `γ`.

Why this works: empirically, the mean subtraction in LayerNorm contributes very little. The dominant effect is the variance scaling. Dropping mean subtraction:
- ~30% faster (one less reduction across the feature dim).
- Slightly fewer parameters.
- Comparable quality.

LLaMA's adoption made RMSNorm the modern default. **For our codebase, swapping `nn.LayerNorm` for `RMSNorm` would be ~5 lines of code, would not change quality meaningfully, and would give a small speedup.** We don't bother because we're reproducing GPT-2 faithfully.

### Why LayerNorm before residual add (pre-LN) — a concrete view

Re-stating the pattern more concretely:

```python
# Pre-LN — what we do:
x = x + attn(ln_1(x))
       │   └────────── attn sees a normalized input
       └────────────── x in the addition is the raw, un-normalized stream

# Post-LN — original 2017 paper:
x = ln_1(x + attn(x))
       │    └────────── attn sees the raw stream
       └─────────────── normalization applied AFTER the residual add
```

In pre-LN, the residual `x + ...` keeps the stream identity-passable: a gradient flowing from layer 12 back to layer 1 can travel through the residual additions without going through any LayerNorm (which would squash its magnitude). In post-LN, every gradient hop crosses a LayerNorm, and through 12 layers this damps gradients to near-zero.

## 4. MLP (the position-wise feedforward)

### What it is

A two-layer MLP applied **independently to each token's vector**. No cross-token interaction — that's attention's job. The MLP just transforms each token's `C`-dim vector into another `C`-dim vector.

Code (`model.py:89-110`):

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1   # see init scheme

    def forward(self, x):
        x = self.c_fc(x)     # (B, T, C) → (B, T, 4C)
        x = self.gelu(x)     # nonlinearity
        x = self.c_proj(x)   # (B, T, 4C) → (B, T, C)
        return x
```

Three steps:

1. **Expand** from `C=768` to `4C=3072` via `c_fc` (the "fully connected" first layer).
2. **Apply GELU** nonlinearity element-wise.
3. **Project back** from `4C=3072` to `C=768` via `c_proj` (the residual stream projection).

The expansion-and-projection-back shape (`C → 4C → C`) is the standard transformer MLP. It's a "bottleneck" architecture turned inside out.

### Why the 4× expansion?

Standard transformer convention. The MLP needs a wide hidden layer to do meaningful nonlinear computation; 4× is empirically a sweet spot. Smaller (2×) underfits; larger (8×) wastes parameters without helping quality much.

The 4× choice ties together: at `n_embd=768`, hidden = 3072. Total MLP params per layer = `768×3072 + 3072 + 3072×768 + 768` ≈ **4.7M params**. With 12 layers = ~57M params in MLPs alone.

That's **~46% of the 124M total**. **MLPs are by far the largest parameter group in the model.** Attention layers are ~28M total (24%). Embeddings (`wte`, tied with `lm_head`) are ~38M (31%). Everything else (LayerNorms, biases) is rounding error.

### Why GELU?

GELU (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) is a smooth approximation of:

```
GELU(x) = x · Φ(x)    where Φ is the standard normal CDF
```

Intuitively: it's like ReLU (`max(0, x)`) but smooth around zero. For large positive `x`, GELU(x) ≈ x. For large negative `x`, GELU(x) ≈ 0. Around 0, it transitions smoothly:

```
       │       /
       │     /
       │    /                  ←  ReLU is a hard hinge at 0
       │   /
       │  / 
─ ─ ─ ─┼─/─ ─ ─ ─
       │/
       /
      /│
     / │
```

GELU smoothly interpolates rather than hard-clipping.

Why GELU over ReLU:
- **Smooth gradients everywhere**, no hard zero region. Helps optimization.
- **Empirically slightly better** than ReLU at the same parameter count on language tasks (small effect, but consistent).
- ReLU's hard cutoff can cause "dead units" — neurons that never activate again because their gradient is exactly zero whenever they're below 0.

GPT-2 used the **tanh approximation** of GELU (because the exact form involves the error function, which was slow on older hardware):

```python
self.gelu = nn.GELU(approximate="tanh")
```

The tanh approximation:
```
GELU_tanh(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

This is what we use (`model.py:100`) to exactly match GPT-2's behavior. Modern hardware can do exact GELU just as fast, so most modern models drop the approximation, but the difference is negligible.

### Modern alternative: SwiGLU

LLaMA, PaLM, Mistral and most post-2023 models use **SwiGLU** (Shazeer 2020) instead of plain GELU. It's structurally different — not just a different activation function:

```python
class GeluMLP(nn.Module):                  class SwiGLUMLP(nn.Module):
    def __init__(self):                        def __init__(self):
        c_fc   = Linear(C, 4*C)                    w1 = Linear(C, ~4*C)
        c_proj = Linear(4*C, C)                    w2 = Linear(C, ~4*C)   ← extra projection
                                                   w3 = Linear(~4*C, C)
    def forward(x):                            def forward(x):
        return c_proj(gelu(c_fc(x)))               return w3(silu(w1(x)) * w2(x))
                                                                ↑
                                                       element-wise gate
```

SwiGLU has three matrices instead of two, and the activation gates one of them by the other. To keep parameter count comparable, the hidden dimension is reduced from `4C` to `~2.67C`.

Why SwiGLU is better:
- The multiplicative gate gives the MLP more expressive power per parameter.
- The Swish/SiLU activation (`x · sigmoid(x)`) is similar to GELU but slightly smoother.
- Empirically beats GELU MLPs by a small but consistent margin.

**For our codebase, swapping in SwiGLU would be ~15 lines and a small quality improvement.** GPT-2 didn't have it (it predates SwiGLU by 2 years), so we don't.

### What MLPs are *for* (interpretability view)

An emerging picture from interpretability research: **MLPs are where the model stores knowledge.** Attention moves information between positions; MLPs apply learned transformations to that information.

A useful frame (Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"): the first matrix `c_fc` projects the input into 3072 "key" directions, each detecting some pattern. The activation gates those keys. The second matrix `c_proj` reads each key's "value" and writes them back to the residual stream. So an MLP is effectively a **soft key-value lookup with 3072 entries.**

This is why MLPs dominate parameter count — they're where most of the model's learned information lives. Attention has structure-and-routing parameters; MLPs have content parameters.

## 5. How they compose: the full block, walkthrough

Walking through `Block.forward` one line at a time, with shapes:

```python
def forward(self, x):
    # x is the residual stream coming in: (B, T, C)
    # We're going to add two things to it: an attention output and an MLP output.

    # === First sublayer: attention ===
    normed_x = self.ln_1(x)
    # normed_x: (B, T, C) — same shape, but each token has been normalized
    # The original x is preserved (it's still the input to this function)

    attn_out = self.attn(normed_x)
    # attn_out: (B, T, C) — attention has now mixed information across positions
    # but the magnitude is bounded because input was normalized

    x = x + attn_out
    # x: (B, T, C) — the stream has been updated:
    # for each token, the attention contribution has been added in
    # (note: x on the LEFT is the un-normalized old x)

    # === Second sublayer: MLP ===
    normed_x = self.ln_2(x)
    # normed_x: (B, T, C) — re-normalized fresh because x just got modified

    mlp_out = self.mlp(normed_x)
    # mlp_out: (B, T, C) — per-token nonlinear transformation, no cross-token info

    x = x + mlp_out
    # x: (B, T, C) — stream updated again

    return x
```

The block is **information-preserving** by construction — `x` enters and exits with the same shape, and the residual additions mean the input is *still recoverable* from the output (you'd just have to invert `attn` and `mlp`, but the information is all there).

This is why the "residual stream" mental model is so useful: every block reads from the stream (via LN + sublayer projections), computes some delta, and writes the delta back. The stream is the persistent state of the model.

## 6. What's modern (post-GPT-2)

Summary of what would change if you wrote this block to modern standards:

| Component | GPT-2 (us) | Modern (LLaMA/Mistral) |
|---|---|---|
| Block structure | Pre-LN ✓ | Pre-LN ✓ (no change) |
| LayerNorm | `nn.LayerNorm` (mean + var) | RMSNorm (RMS only) |
| Attention | MHA | GQA (see `attention-variants.md`) |
| MLP activation | GELU (tanh approx) | SiLU / Swish |
| MLP structure | 2 matrices, 4× expansion | SwiGLU: 3 matrices, ~2.67× expansion |
| Positional encoding | Learned `wpe` added to embeddings | RoPE (rotates Q/K inside attention) |
| Bias terms | Yes everywhere | Most modern models drop biases |
| Final LN | `ln_f` ✓ | RMSNorm in same place |

**Bias terms** is one I haven't called out yet. GPT-2 puts `bias=True` on all Linear layers (default). LLaMA and most modern models set `bias=False` everywhere — saves a tiny number of params and a tiny amount of compute, and empirically doesn't hurt quality. Probably the cheapest "modern" tweak you could make to this codebase.

## 7. Why this design works

A high-level summary of why the modern transformer block (pre-LN, residual, LayerNorm, MLP, attention) is so effective:

1. **Residual connections** let gradients flow through arbitrarily deep stacks without vanishing.
2. **Pre-LN** keeps sublayer inputs at a well-behaved scale without breaking the residual path.
3. **LayerNorm** is independent of batch size, sequence length, train/eval mode — so it scales cleanly.
4. **The residual stream** as a shared, additive workspace lets later layers build on earlier ones without explicitly being told to.
5. **Per-position MLPs** do the heavy parameter lifting; attention does the cross-position routing. **Clean separation of "what" (MLP) and "where" (attention).**
6. **GELU/SwiGLU** provides smooth gradients with enough nonlinearity for the MLP to do meaningful computation.

None of these pieces is irreplaceable in isolation. The combination is what works. Architectural papers have tried changing each component individually for years; the best swap-ins are usually small wins (RMSNorm, SwiGLU, RoPE). The big picture has been stable since GPT-2.
