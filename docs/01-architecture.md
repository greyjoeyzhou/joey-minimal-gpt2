# Architecture: GPT-2 124M, Block by Block

This document walks through `model.py` line by line at a conceptual level. The
code itself has line-by-line comments — read both together.

## The high-level shape

A decoder-only transformer with 12 layers. Each token in (B, T) goes through:

    embed -> 12x [LN -> Attention -> +residual -> LN -> MLP -> +residual] -> LN -> lm_head -> logits

That's it. No encoder, no cross-attention, no recurrence.

## Embeddings (wte + wpe)

- `wte`: token embedding table, (vocab_size=50304, n_embd=768).
- `wpe`: learned positional embedding table, (block_size=1024, n_embd=768).
- The model sees: `tok_emb + pos_emb` at each position.

We use *learned* absolute positional embeddings, matching the original GPT-2.
Modern best practices favor RoPE (rotary) or ALiBi for length generalization,
but GPT-2 used learned absolute. We follow the paper.

## The transformer block (pre-LN)

```
x_in
 |
 +--+
 |  |
 |  LN_1
 |  |
 |  Attention (causal)
 |  |
 +<-+   <- residual add
 |
 +--+
 |  |
 |  LN_2
 |  |
 |  MLP (4x expansion + GELU)
 |  |
 +<-+   <- residual add
 |
x_out
```

Key idea: the *residual stream* (the path going straight down) passes through
every block unchanged; sublayers add to it. This view is the basis for
mechanistic interpretability work.

Pre-LN vs post-LN: the original transformer paper applied LN *after* the
residual add. GPT-2 onward applies LN *before* each sublayer (inside the
residual branch, not on the output). Pre-LN makes very deep transformers
trainable without aggressive learning-rate tricks.

## Causal self-attention

For each head:
    Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
    A = softmax(Q K^T / sqrt(d_k) + causal_mask)
    out = A V
    out projected through W_o

We compute Q, K, V in a single fused linear (`c_attn`), then split.

`F.scaled_dot_product_attention(q, k, v, is_causal=True)` is PyTorch's
dispatch into the best available kernel — on the 5090 with bf16, this gives
us flash attention with O(N) memory.

Multi-head: we split the embedding into 12 heads of dim 64 each. Each head
attends independently; outputs are concatenated and projected by `c_proj`.

## MLP

Two linear layers with a 4x expansion: 768 -> 3072 -> 768. Activation is GELU
(tanh approximation, matching GPT-2). This is the bulk of the compute and
parameters: each layer's MLP holds ~7.1M params vs ~1.8M for its attention.

## LM head + weight tying

The final LN's output is projected to vocab logits by `lm_head`. The trick:
`lm_head.weight is wte.weight` — the same tensor. This saves ~38M params at
this scale (vocab × d_model) and tends to slightly improve quality.

## Init scheme

- Linear/Embedding weights: normal(0, 0.02).
- Biases / LN params: zero / one (PyTorch defaults).
- `c_proj` (the residual stream projection inside attention and MLP) gets a
  *scaled* init: std = 0.02 / sqrt(2 * n_layer). This keeps the residual
  stream variance roughly constant across depth.

## Parameter count math (≈ 124M)

- wte: 50304 × 768 = 38.6M
- wpe: 1024 × 768 = 0.79M
- Per block: 3*768*768 (qkv) + 768*768 (c_proj) + 768*3072 + 3072*768 (mlp)
        + 2*(768*2) (LN params) ≈ 7.08M
- 12 blocks: 85M
- Final LN + lm_head: lm_head is tied to wte, so no new params.
- Total: ~38.6 + 0.79 + 85 ≈ 124M

The 50304 padded vocab adds ~36k params over the real 50257 — negligible.
