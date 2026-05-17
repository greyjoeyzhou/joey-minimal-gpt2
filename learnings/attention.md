# Attention

Notes on the attention mechanism in our GPT-2 implementation — what it does conceptually, the QKV math step-by-step, a naive PyTorch implementation that exposes what `F.scaled_dot_product_attention` does internally, a worked numerical example, why each design choice is the way it is, the trainable parameters, how our production code maps to the naive version, and how cross-attention generalizes the mechanism beyond self-attention.

See also: [`embedding.md`](./embedding.md) (token/position vectors that feed into attention), [`abbreviations.md`](./abbreviations.md) (every name you'll see in `model.py`).

## 1. What attention does

Attention is the mechanism that lets every token in a sequence look at every other token and decide *which ones are relevant to me right now*. It puts the "T" in "transformer" — **the only place in the whole model where information flows between positions.** The MLP processes each position independently. Embeddings are per-token. Attention is the lone cross-position operation.

### The intuition — query, key, value

Each token wants to ask: *"Of all the previous tokens, which ones should I pay attention to in order to predict what comes next?"*

The mechanism gives each token three projections of itself:

- **Query (Q)** — "here's what I'm looking for"
- **Key (K)** — "here's what I have to offer / how I should be indexed"
- **Value (V)** — "here's the actual content I'd hand over if you matched my key"

For each token:

1. Take its Q vector.
2. Dot-product it with every other token's K vector. High dot product = high relevance.
3. Softmax those scores into a probability distribution (attention weights).
4. Weighted sum of all V vectors using those weights.

The output is a custom blend of every other token's "value" content, weighted by relevance.

### The one-line math

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

`d_k` is the per-head Q/K dimension (`head_dim` = 64 for us). The `√d_k` divisor keeps the softmax numerically well-behaved.

### Causal — the twist for language modeling

For autoregressive LMs: **token at position t can only attend to positions 0..t, not t+1..T**. Otherwise the model would cheat (predict token 5 by looking at token 7). Enforced by setting attention scores for "future" positions to `-∞` before the softmax, making their weights zero.

### Multi-head — parallel specialization

Instead of one big attention computation with full 768-dim Q/K/V, we split into **12 heads of 64 dims each**. Each head does independent Q/K/V projections, dot products, softmax, weighted sum — then concatenate back to 768. Different heads can specialize in different relationships (syntactic, semantic, positional). Multi-head consistently beats single-head at the same param count.

## 2. The mechanism — 9 steps

For input `x` of shape `(B, T, C)`:

```
x         (B, T, C)        ← input: T tokens, each a C-dim vector

(1) project to Q, K, V:
    Q = x @ W_q + b_q       (B, T, C)
    K = x @ W_k + b_k       (B, T, C)
    V = x @ W_v + b_v       (B, T, C)

(2) reshape for multi-head:
    Q, K, V                  (B, n_head, T, head_dim)   where head_dim = C/n_head

(3) attention scores:
    S = Q @ Kᵀ              (B, n_head, T, T)
                              S[b, h, i, j] = how strongly token i attends to token j

(4) scale:
    S = S / √head_dim       prevents softmax saturation in high dimensions

(5) causal mask:
    S[i, j] = -∞  for j > i  prevents attending to future tokens

(6) softmax along j:
    A = softmax(S, dim=-1)  (B, n_head, T, T)
                              each row sums to 1 — probability distribution

(7) weighted sum of values:
    O = A @ V               (B, n_head, T, head_dim)

(8) reshape back:
    O                        (B, T, C)

(9) output projection:
    out = O @ W_o + b_o     (B, T, C)
```

## 3. Naive implementation

This produces identical output to `CausalSelfAttention` in `model.py:35-86`. The only difference: the fused-kernel call is replaced with explicit ops.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveCausalSelfAttention(nn.Module):
    """Self-attention written out longhand. Same output as model.py's version.

    Differences from production code:
      - Three separate Q/K/V projections instead of fused c_attn (clearer to read)
      - Explicit attention computation instead of F.scaled_dot_product_attention
        (slower but shows what's happening)
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0

        self.W_q = nn.Linear(n_embd, n_embd)
        self.W_k = nn.Linear(n_embd, n_embd)
        self.W_v = nn.Linear(n_embd, n_embd)
        self.W_o = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Precompute the causal mask as a lower-triangular matrix of 1s.
        # Registered as a buffer so it moves to GPU with the module but
        # isn't trained.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # --- (1) Project to Q, K, V ---
        # Each Linear takes (B, T, C) and returns (B, T, C).
        # Three independent learned linear transformations of the same input.
        q = self.W_q(x)  # (B, T, C)
        k = self.W_k(x)  # (B, T, C)
        v = self.W_v(x)  # (B, T, C)

        # --- (2) Reshape for multi-head ---
        # Split the C dim into n_head heads of head_dim each.
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- (3) Attention scores: Q @ Kᵀ ---
        # For each head, compute dot products between every query and every key.
        # k.transpose(-2, -1):  (B, nh, T, hd) -> (B, nh, hd, T)
        # q @ k.transpose(-2, -1):  (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T)
        scores = q @ k.transpose(-2, -1)  # (B, nh, T, T)

        # --- (4) Scale ---
        # Without scaling, dot products of length-d vectors with N(0,1) entries
        # have variance d, so they grow with dimensionality. Softmax of large
        # numbers saturates to one-hot (gradient becomes zero).
        scores = scores / (self.head_dim ** 0.5)

        # --- (5) Causal mask ---
        # Set positions where j > i to -∞ so softmax assigns them weight 0.
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf"),
        )

        # --- (6) Softmax ---
        # Normalize across the "what to attend to" dim. Each row sums to 1.
        attn = F.softmax(scores, dim=-1)  # (B, nh, T, T)

        # --- (7) Weighted sum of values ---
        # (B, nh, T, T) @ (B, nh, T, hd) -> (B, nh, T, hd)
        out = attn @ v

        # --- (8) Reshape back ---
        # (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # --- (9) Output projection ---
        # Re-mix information across heads (the head split was a computational
        # trick; we re-mix so heads aren't isolated from the next layer).
        out = self.W_o(out)
        return out
```

## 4. A worked numerical example

Tiny case to walk through by hand: **B=1, T=3, 1 head, head_dim=2**, so `n_embd=2`. Three tokens, each a 2-dim vector.

For maximum clarity, pretend `W_q`, `W_k`, `W_v` are identity matrices (so `Q = K = V = x`).

```
x = [[1, 0],     # token 0
     [0, 1],     # token 1
     [1, 1]]     # token 2

Q = K = V = x
```

**Step 3 — attention scores (Q @ Kᵀ):**

```
S[i,j] = q_i · k_j

S = [[1·1+0·0,  1·0+0·1,  1·1+0·1],     [[1, 0, 1],
     [0·1+1·0,  0·0+1·1,  0·1+1·1],  =   [0, 1, 1],
     [1·1+1·0,  1·0+1·1,  1·1+1·1]]      [1, 1, 2]]
```

Geometrically: each entry is the dot product (similarity) between query i and key j.

**Step 4 — scale by √2 ≈ 1.41:**

```
S = [[0.71, 0.00, 0.71],
     [0.00, 0.71, 0.71],
     [0.71, 0.71, 1.41]]
```

**Step 5 — causal mask (everything above diagonal → -∞):**

```
S = [[ 0.71,  -inf,  -inf],
     [ 0.00,  0.71,  -inf],
     [ 0.71,  0.71,  1.41]]
```

**Step 6 — row-wise softmax:**

Row 0: only one finite entry → `[1.00, 0, 0]`

Row 1: softmax of `[0, 0.71]`:
- e⁰ = 1, e⁰·⁷¹ ≈ 2.03, sum = 3.03
- → `[0.33, 0.67, 0]`

Row 2: softmax of `[0.71, 0.71, 1.41]`:
- e⁰·⁷¹ ≈ 2.03, e⁰·⁷¹ ≈ 2.03, e¹·⁴¹ ≈ 4.10, sum = 8.16
- → `[0.25, 0.25, 0.50]`

So:

```
A = [[1.00,  0.00,  0.00],     ← token 0: 100% on itself
     [0.33,  0.67,  0.00],     ← token 1: 33% on tok 0, 67% on itself
     [0.25,  0.25,  0.50]]     ← token 2: spread across all, mostly itself
```

**Step 7 — weighted sum of values (A @ V):**

```
out[0] = 1.00·[1,0] + 0·[0,1] + 0·[1,1]       = [1.00, 0.00]
out[1] = 0.33·[1,0] + 0.67·[0,1] + 0·[1,1]    = [0.33, 0.67]
out[2] = 0.25·[1,0] + 0.25·[0,1] + 0.50·[1,1] = [0.75, 0.75]
```

Each output row is a **weighted blend of value vectors**, where the weights came from query-key similarity. Token 2's output `[0.75, 0.75]` is a soft mix of all three input tokens — that's information flowing across positions.

In the real model, the weights wouldn't be these uniform-looking values because of the learned `W_q`/`W_k`/`W_v` warping the geometry — the model learns *which* directions in embedding space should correspond to "queries that match this key." That learning is the whole point.

## 5. Why each step

**Why Q, K, V instead of just one matrix?** Three independent linear projections give the model the flexibility to decouple the roles. If Q = K, every token would attend most to itself (perfect self-similarity). If K = V, the "addressing" and "payload" would be coupled — couldn't have a key that says "I'm a noun" and a value that delivers "the embedded meaning of the noun."

**Why dot product?** Natural similarity measure in continuous space, composes with linear algebra (matmul). Early attention papers tried additive attention (`v · tanh(W_q q + W_k k)`); dot-product won on speed and GPU-friendliness.

**Why scale by √head_dim?** If Q and K entries are roughly N(0,1), a dot product of d-dim vectors has variance d, so scores grow as √d. Softmax of large numbers saturates to one-hot (gradient through softmax → 0). Dividing by √d keeps softmax inputs at variance ~1. **Without this, training breaks at larger model sizes.** For our `head_dim=64`, scores would have std ≈ 8 unscaled, ≈ 1 scaled — huge difference for softmax.

**Why softmax?** (1) Need a probability distribution (non-negative, sums to 1) for the weighted average. (2) Smooth and differentiable, gradients flow back cleanly. Alternatives (top-k, ReLU-then-normalize) have been tried; softmax stays standard.

**Why causal mask?** Autoregressive LMs predict each token from previous ones only. Without the mask, token 5 could attend to token 7 and trivially learn to copy it — 100% accuracy, zero learning. The mask also makes **one forward pass equivalent to T separate next-token predictions** — every position simultaneously gets trained on its own next-token target.

**Why multi-head?** Single big head means one softmax decision per position. Multiple smaller heads = parallel diverse decisions, each potentially focusing on different relationships. Cost: each head has lower-dim Q/K → less precise individual attention. Benefit usually outweighs cost; 8-32 heads is standard.

**Why output projection (`W_o`)?** After multi-head split, each head occupies a fixed slice of the output dim (head 0 → dims 0-63, head 1 → 64-127, ...). Without `W_o`, the next layer sees information siloed by head, with no mixing. `W_o` is a learned linear that re-mixes across heads.

## 6. The trainable parameters

Q, K, V are all linear projections of the same `x`, using **separate trainable weight matrices**. These are the only learned things inside an attention layer — the softmax, scaling, masking, matmuls are all fixed math. What gets learned is **the linear transformations that decide what aspect of each token's representation serves as a query, key, or value.**

Per attention layer (with our defaults of `n_embd=768`):

| Matrix | Shape | Params |
|---|---|---|
| `W_q` | (768, 768) | 590K + 768 bias |
| `W_k` | (768, 768) | 590K + 768 bias |
| `W_v` | (768, 768) | 590K + 768 bias |
| `W_o` | (768, 768) | 590K + 768 bias |
| **Total per layer** | | **~2.36M params** |

With 12 layers, attention alone accounts for ~28M parameters — roughly a quarter of the 124M total.

## 7. How our actual code maps to this naive version

| Naive step | Production code (`model.py:35-86`) | Why it's different |
|---|---|---|
| `W_q(x), W_k(x), W_v(x)` (3 linears) | `c_attn(x).split(...)` (1 fused linear) | One matmul is faster than three. Math identical. |
| Reshape to multi-head | Same | — |
| Explicit `Q @ Kᵀ`, scale, mask, softmax, `@ V` | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` | PyTorch's kernel fuses all into one GPU op. On modern hardware dispatches to **Flash Attention** — same math, reordered to fit the GPU memory hierarchy. 2-5× speedup, lower memory. |
| Reshape back | Same | — |
| `W_o(out)` | `self.c_proj(y)` | Different name only |

The math is identical. The production version is numerical equivalence wrapped in two optimizations: fused projection (small matmul savings) and Flash Attention (big savings).

The fused QKV trick is just stacking the three weight matrices horizontally:

```python
# model.py:51 — instead of three separate Linears, one fat one:
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
# Shape: (768, 2304) — that's W_q, W_k, W_v concatenated along output dim
```

```python
# model.py:68-69 — apply it, then slice:
qkv = self.c_attn(x)                       # (B, T, 3*C)
q, k, v = qkv.split(self.n_embd, dim=2)    # three (B, T, C) tensors
```

PyTorch's autograd treats `c_attn.weight` as a single parameter, but conceptually it's three independent projections — gradients flowing back update the three "thirds" independently. Same trainable structure, same expressive power, one CUDA kernel instead of three.

## 8. Cross-attention — when Q comes from elsewhere

Self-attention is *"attend from x to x."* **Cross-attention** is *"attend from sequence A to sequence B."* Same mechanism, different sources for Q vs K/V.

### The canonical use case: encoder-decoder translation

The original 2017 "Attention Is All You Need" paper was about English-to-German translation. The decoder had **three** sublayers per block, not two:

```
ENCODER (English source)              DECODER (German target)
    ┌─────────────────┐                ┌─────────────────┐
    │ Self-Attention  │                │ Causal Self-Attn│
    │  (bidirectional)│                │       ↓         │
    │       ↓         │                │ CROSS-ATTENTION │ ← Q from decoder,
    │     MLP         │ ──encoded──►  │   Q from decoder,│   K,V from encoder
    │       ↓         │     source     │   K,V from enc  │
    │     ...         │                │       ↓         │
    │   final state ──┴── K,V into ──► │     MLP         │
    └─────────────────┘   every layer's└─────────────────┘
                          cross-attn
```

In the cross-attention sublayer, the decoder asks: "given what I've generated in German so far, which parts of the English source should I look at?"

### The structural difference, in code

```python
class CrossAttention(nn.Module):
    """Cross-attention: Q from sequence A, K/V from sequence B.

    A and B can have DIFFERENT lengths (T_q vs T_kv).
    Typically NO causal mask — full B is visible to every position in A.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.W_q = nn.Linear(n_embd, n_embd)   # Q from sequence A
        self.W_k = nn.Linear(n_embd, n_embd)   # K from sequence B
        self.W_v = nn.Linear(n_embd, n_embd)   # V from sequence B
        self.W_o = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.head_dim = n_embd // n_head

    def forward(self, x_q, x_kv):
        """
        x_q:  (B, T_q,  C)   "queries" — typically decoder's current state
        x_kv: (B, T_kv, C)   "context" — typically encoder's output
        """
        B, T_q, C = x_q.shape
        T_kv = x_kv.size(1)

        # Q from one source, K and V from the other:
        q = self.W_q(x_q)   # (B, T_q,  C)
        k = self.W_k(x_kv)  # (B, T_kv, C)
        v = self.W_v(x_kv)  # (B, T_kv, C)

        # Reshape for multi-head (same as self-attention)
        q = q.view(B, T_q,  self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: (B, nh, T_q, T_kv) — RECTANGULAR shape!
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # NO causal mask — every query can see every key
        attn = F.softmax(scores, dim=-1)
        out = attn @ v   # (B, nh, T_q, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.W_o(out)
```

Three structural differences from self-attention:

| | Self-attention | Cross-attention |
|---|---|---|
| Sources of Q, K, V | All from same input | Q from one source, K/V from another |
| Attention matrix shape | `(T, T)` — square | `(T_q, T_kv)` — rectangular |
| Causal mask? | Yes (in decoder LMs) | Usually no (full context visible) |

**Sequence lengths can differ** — translating a 20-word English sentence into 24 words of German? The decoder's T_q grows from 1 to 24 as it generates, but T_kv is always 20 (fixed encoded source). The attention matrix is `24 × 20` at the end.

### Where cross-attention lives in modern systems

Pure-language decoder-only models (GPT, LLaMA, Mistral, Qwen, Claude) **don't use cross-attention**. Everything is causal self-attention; source and target are just concatenated into one long sequence:

```
[system prompt] [user message] [assistant message so far]
                                    ↑ generating from here
```

All causal self-attention. Simpler, more flexible, empirically just as good or better. This is why GPT-style decoder-only architectures won out over encoder-decoder for general LMs.

But cross-attention is **everywhere in multimodal models** — wherever you have two different "kinds" of input:

| Model / family | What attends to what |
|---|---|
| **Original Transformer / T5 / BART** | German decoder → English encoder (translation) |
| **Whisper** (OpenAI's ASR) | Text decoder → audio encoder spectrogram features |
| **Stable Diffusion / DALL-E 2** | Image U-Net features → text encoder (CLIP) embeddings |
| **Flamingo** (DeepMind) | LM layers → vision encoder features (interleaved cross-attn) |
| **BLIP / IDEFICS** | Language decoder → image patch tokens |
| **LLaVA** | Doesn't use cross-attention — projects vision into LM token space and uses pure self-attention |
| **Perceiver / Perceiver IO** | Small latent "queries" → arbitrary big input (pixels, audio, point clouds) |
| **RETRO** (DeepMind) | LM decoder → retrieved external documents (retrieval-augmented LM) |
| **SAM** (Meta segmentation) | Prompt tokens (clicks, boxes) → image features |

Cross-attention is the **bridge between modalities** (text↔image, text↔audio) or between structured roles (decoder↔encoder, query↔retrieved-context).

### Design tension in modern multimodal

Two ways to wire vision into a language model:

**(a) Cross-attention** (Flamingo-style): Keep the LM unchanged, add cross-attention layers between the LM's self-attention layers. Cross-attention is the *only* place vision enters. Allows pretraining the LM independently. More parameters added per layer.

**(b) Concatenate-and-self-attend** (LLaVA-style): Project image patches into LM-shaped vectors, prepend to text tokens, run normal causal self-attention. Image tokens look the same as text to the model. Simpler. Reuses the entire decoder-only LM machinery unchanged.

(b) has been winning recently for cost and simplicity reasons — same pattern as how decoder-only beat encoder-decoder for pure language. **Architectural specialization tends to lose to "treat everything as one sequence."**

Cross-attention is still essential when modalities have very different scales — a Perceiver looking at a million pixels with a 256-dim latent space, where concatenating into the LM's sequence would blow up the context length.

### For our codebase

We have **zero cross-attention** in this repo. GPT-2 is decoder-only; everything is causal self-attention. But the cross-attention shape is worth knowing because:

- It's the natural generalization "attend from A to B."
- If you ever extend this codebase toward multimodal, you'll add cross-attention modules.
- Modern attention variants (KV cache, GQA, MLA) all carry over to cross-attention — the K/V optimization tricks work the same regardless of whether K/V come from the same sequence as Q.
