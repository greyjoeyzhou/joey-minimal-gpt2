"""Nanowhale-inspired transformer (v2).

Adds four frontier techniques on top of model_v1.py (RMSNorm + RoPE + SwiGLU + GQA):

  1. MLA  — Multi-head Latent Attention. Compresses K/V into a low-rank latent
            vector, drastically shrinking the inference KV cache. Also splits
            each head into RoPE dims (position-aware) and NoPE dims (content-only).
  2. MoE  — Mixture of Experts. Replaces the dense FFN with a sparse bank of
            experts: n_shared_experts always fire, top-k of n_routed_experts are
            selected per token by a learned router.
  3. Hyper-Connections — replaces the standard residual `x += f(x)` with a
            multi-stream hidden state where input/output weights are learned per
            layer, allowing richer skip patterns across depth.
  4. MTP  — Multi-Token Prediction. An auxiliary head that predicts token t+2
            alongside the main t+1 prediction, encouraging the model to represent
            multiple future tokens in its hidden states.

Architecture comparison:

  | Subsystem       | v1 (modern)         | v2 (nanowhale-style)              |
  |-----------------|---------------------|-----------------------------------|
  | Attention       | GQA                 | MLA (low-rank latent KV + RoPE/NoPE split) |
  | FFN             | Dense SwiGLU        | Sparse MoE (shared + routed experts)       |
  | Residual        | Standard x += f(x)  | Hyper-Connections (multi-stream)            |
  | Prediction      | Next token only     | Next token + MTP auxiliary head             |
  | Layers          | 12                  | 8 (but MoE multiplies capacity)             |

References:
  MLA             https://arxiv.org/abs/2405.04434 (DeepSeek-V2)
  MoE             https://arxiv.org/abs/2101.03961 (Switch Transformer)
  Hyper-Connections https://arxiv.org/abs/2409.19606
  MTP             https://arxiv.org/abs/2404.19737 (DeepSeek-V3)
  nanowhale       https://huggingface.co/HuggingFaceTB/nanowhale-100m-base
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v2 import GPTConfig


# ---------------------------------------------------------------------------
# RoPE helpers  (identical to v1 — only precomputed for rope_head_dim, not full head_dim)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(rope_head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex RoPE phasors for the rope portion of each head.

    Identical to v1 but operates on rope_head_dim (e.g. 32) instead of the
    full head_dim (e.g. 96). The NoPE dimensions are left unrotated.

    Returns: complex tensor of shape (seq_len, rope_head_dim // 2).
    """
    assert rope_head_dim % 2 == 0
    inv_freqs = 1.0 / (theta ** (torch.arange(0, rope_head_dim, 2).float() / rope_head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, inv_freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate a Q or K tensor by the precomputed RoPE phasors.

    Args:
        x:         (B, n_head, T, rope_head_dim) — the rope portion only.
        freqs_cis: (T, rope_head_dim // 2) complex.
    Returns: rotated tensor, same shape and dtype as x.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(3).to(x.dtype)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Normalization — same as v1."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_inv = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms_inv * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# 1. MLA — Multi-head Latent Attention
# ---------------------------------------------------------------------------

class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, 2024).

    Standard GQA caches K and V at full resolution: n_kv_head * head_dim per
    token per layer. MLA instead compresses K and V through a shared low-rank
    latent vector c_KV of dimension kv_lora_rank << n_kv_head * head_dim.

    At inference, the KV cache stores:
        c_KV  (kv_lora_rank dims) — enough to reconstruct K_nope and V.
        k_rope (n_kv_head * rope_head_dim dims) — the positional part.

    Cache size ratio vs. MHA:
        MHA:  2 * n_head * head_dim  = 2 * 8 * 96 = 1536 per token
        MLA:  kv_lora_rank + n_kv_head * rope_head_dim = 96 + 32 = 128 per token
        → ~12× smaller KV cache.

    Head structure per head:
        NoPE dims (nope_head_dim=64): content features, no positional rotation.
        RoPE dims (rope_head_dim=32): positional features, rotated by RoPE.
        head_dim = rope_head_dim + nope_head_dim = 96.

    Q path (with LoRA compression via q_lora_rank):
        c_Q      = W_dq(x)          : (B, T, q_lora_rank)
        q_nope   = W_uq_nope(c_Q)   : (B, T, n_head * nope_head_dim)
        q_rope   = W_uq_rope(c_Q)   : (B, T, n_head * rope_head_dim)
        Q        = cat(RoPE(q_rope), q_nope) per head

    KV path (shared latent):
        c_KV     = W_dkv(x)         : (B, T, kv_lora_rank)
        k_nope   = W_uk_nope(c_KV)  : (B, T, n_kv_head * nope_head_dim)
        v        = W_uv(c_KV)       : (B, T, n_kv_head * head_dim)
        k_rope   = W_kr(x)          : (B, T, n_kv_head * rope_head_dim)  ← from x, not c_KV
        K        = cat(RoPE(k_rope), k_nope) per head

    K and V are then expanded from n_kv_head=1 to n_head=8 (like GQA).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head       = config.n_head
        self.n_kv_head    = config.n_kv_head
        self.head_dim     = config.rope_head_dim + config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.nope_head_dim
        self.n_rep        = config.n_head // config.n_kv_head  # expansion ratio

        # Q path: down-project to q_lora_rank, then up-project to rope and nope parts.
        self.q_down  = nn.Linear(config.n_embd, config.q_lora_rank,                        bias=False)
        self.q_up_nope = nn.Linear(config.q_lora_rank, config.n_head * config.nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(config.q_lora_rank, config.n_head * config.rope_head_dim, bias=False)

        # KV path: shared latent down-projection, separate up-projections.
        self.kv_down   = nn.Linear(config.n_embd,      config.kv_lora_rank,                        bias=False)
        self.k_up_nope = nn.Linear(config.kv_lora_rank, config.n_kv_head * config.nope_head_dim,   bias=False)
        self.v_up      = nn.Linear(config.kv_lora_rank, config.n_kv_head * self.head_dim,           bias=False)

        # K_rope is computed directly from x (not from c_KV) — this is the
        # part that must be cached alongside c_KV at inference.
        self.k_rope    = nn.Linear(config.n_embd, config.n_kv_head * config.rope_head_dim, bias=False)

        # Output projection: n_head * head_dim -> n_embd.
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.o_proj.SCALE_INIT = 1  # type: ignore[attr-defined]

        # RMSNorm on the KV latent (stabilizes the low-rank projection).
        self.kv_norm = RMSNorm(config.kv_lora_rank)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # --- Q ---
        c_Q      = self.q_down(x)                                   # (B, T, q_lora_rank)
        q_nope   = self.q_up_nope(c_Q)                              # (B, T, n_head * nope_head_dim)
        q_rope   = self.q_up_rope(c_Q)                              # (B, T, n_head * rope_head_dim)
        q_nope   = q_nope.view(B, T, self.n_head, self.nope_head_dim).transpose(1, 2)
        q_rope   = q_rope.view(B, T, self.n_head, self.rope_head_dim).transpose(1, 2)
        q_rope   = apply_rope(q_rope, freqs_cis)
        # Concatenate: RoPE dims first (matches K ordering below).
        q = torch.cat([q_rope, q_nope], dim=-1)                     # (B, n_head, T, head_dim)

        # --- K, V ---
        c_KV     = self.kv_norm(self.kv_down(x))                    # (B, T, kv_lora_rank)
        k_nope   = self.k_up_nope(c_KV)                             # (B, T, n_kv_head * nope_head_dim)
        v        = self.v_up(c_KV)                                  # (B, T, n_kv_head * head_dim)
        k_rope_raw = self.k_rope(x)                                  # (B, T, n_kv_head * rope_head_dim)
        k_nope   = k_nope.view(B, T, self.n_kv_head, self.nope_head_dim).transpose(1, 2)
        k_rope_t = k_rope_raw.view(B, T, self.n_kv_head, self.rope_head_dim).transpose(1, 2)
        k_rope_t = apply_rope(k_rope_t, freqs_cis)
        k = torch.cat([k_rope_t, k_nope], dim=-1)                   # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Expand K, V from n_kv_head to n_head (same as GQA).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# 2. MoE — Mixture of Experts
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single SwiGLU feedforward expert."""

    def __init__(self, n_embd: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.up_proj   = nn.Linear(n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, n_embd, bias=False)
        self.down_proj.SCALE_INIT = 1  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts feedforward layer.

    Architecture:
        n_shared_experts:  always process every token (guaranteed capacity).
        n_routed_experts:  compete via a learned router; only top-k fire per token.
        output = shared_output + weighted_sum(top-k routed expert outputs)

    Why shared experts?
        Pure routing can leave some tokens underserved if all their preferred
        experts are at capacity. Shared experts guarantee a baseline signal for
        every token regardless of routing.

    Router:
        A single linear layer maps each token vector to n_routed_experts logits.
        Softmax + top-k selects which experts activate. The selected weights are
        renormalized to sum to 1 before weighting expert outputs.

    Load balance loss (Switch Transformer, Fedus et al. 2022):
        Without regularization, routers collapse to always using the same 1-2
        experts. The auxiliary loss penalizes imbalance:

            L_aux = router_scale * n_experts * Σ_i (f_i * P_i)

        where f_i = fraction of tokens dispatched to expert i (discrete),
              P_i = mean router probability for expert i (differentiable).

        f * P is a differentiable proxy for "expert i gets too many tokens":
        when f_i is high (many tokens routing here), making P_i small
        reduces the loss, which pushes the router away from that expert.

    Efficiency note:
        The loop below processes each expert on its subset of tokens. For
        production use, expert-parallel dispatch + gather kernels (e.g.
        Megablocks) replace this loop with vectorized CUDA ops.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_routed    = config.n_routed_experts
        self.n_shared    = config.n_shared_experts
        self.top_k       = config.n_experts_per_tok
        self.aux_scale   = config.router_scale

        self.shared_experts = nn.ModuleList([
            Expert(config.n_embd, config.moe_intermediate)
            for _ in range(config.n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(config.n_embd, config.moe_intermediate)
            for _ in range(config.n_routed_experts)
        ])
        # Router: one linear, no bias, maps token -> expert scores.
        self.router = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out:     (B, T, n_embd) — combined expert output.
            lb_loss: scalar load-balance auxiliary loss.
        """
        B, T, C = x.shape

        # --- Shared experts (all tokens) ---
        shared_out = sum(e(x) for e in self.shared_experts)

        # --- Router ---
        # scores: (B, T, n_routed) — softmax probabilities over routed experts.
        scores = F.softmax(self.router(x), dim=-1)

        # Top-k selection per token.
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        # Renormalize selected weights so they sum to 1 (avoids scale sensitivity).
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # --- Routed experts ---
        # Flatten batch+seq into a single token dimension for dispatch.
        flat_x      = x.view(B * T, C)
        flat_idx    = topk_idx.view(B * T, self.top_k)
        flat_scores = topk_scores.view(B * T, self.top_k)

        routed_out = torch.zeros(B * T, C, device=x.device, dtype=x.dtype)
        for e_id, expert in enumerate(self.routed_experts):
            # For each top-k slot, find tokens that chose this expert.
            for k in range(self.top_k):
                mask = (flat_idx[:, k] == e_id)   # (B*T,) boolean
                if not mask.any():
                    continue
                w    = flat_scores[mask, k].unsqueeze(-1)  # (n_selected, 1)
                out  = expert(flat_x[mask])                # (n_selected, C)
                routed_out[mask] += w * out

        routed_out = routed_out.view(B, T, C)

        # --- Load balance loss ---
        lb_loss = self._load_balance_loss(scores, topk_idx)

        return shared_out + routed_out, lb_loss

    def _load_balance_loss(
        self, scores: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """Switch Transformer auxiliary load-balance loss.

        L = router_scale * n_experts * sum_i(f_i * P_i)

        f_i: fraction of tokens dispatched to expert i (non-differentiable).
        P_i: mean softmax router probability for expert i (differentiable).
        Their product is a differentiable surrogate for load imbalance.
        """
        n = self.n_routed
        # P_i: mean routing probability per expert — differentiable.
        P = scores.mean(dim=(0, 1))                               # (n_routed,)
        # f_i: fraction of top-k slots assigned to each expert.
        one_hot = torch.zeros(*scores.shape, device=scores.device)
        one_hot.scatter_(-1, topk_idx, 1.0)                      # (B, T, n_routed)
        f = one_hot.mean(dim=(0, 1)) / self.top_k                 # (n_routed,)
        return self.aux_scale * n * (f * P).sum()


# ---------------------------------------------------------------------------
# 3. Block with Hyper-Connections
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block with Hyper-Connections (Zhu et al., 2024).

    Standard residual:
        x = x + attn(norm(x))
        x = x + mlp(norm(x))

    Hyper-Connections maintain hc_expansion parallel "streams" of the hidden
    state rather than a single one. Each sublayer (attn, moe):
      1. Reads a learned weighted combination of all streams as its input.
      2. Distributes its output back to all streams with learned per-stream weights.

    This generalizes the residual connection: stream 0 behaves like the standard
    residual path, while extra streams can learn to act as skip connections that
    bypass groups of layers, carry specialized features, etc.

    Initialization:
        alpha (input weights): heavily weighted toward stream 0 → nearly identical
            to standard residual at the start of training.
        beta  (output weights): sigmoid(5) ≈ 0.99 for stream 0, sigmoid(-5) ≈ 0 for
            others → output flows almost entirely to stream 0 initially.

    Memory: the hidden state h has shape (B, T, hc_expansion, n_embd) — hc_expansion×
    larger than a standard transformer. With hc_expansion=2 and n_embd=768 this is
    manageable; nanowhale uses hc_expansion=4 with n_embd=320 for similar memory.

    Interface:
        forward(h, freqs_cis) -> (h_updated, lb_loss)
        h has shape (B, T, hc_expansion, n_embd).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hc = config.hc_expansion

        self.rms_1 = RMSNorm(config.n_embd)
        self.attn  = MLA(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.moe   = MoELayer(config)

        # Learnable input (alpha) and output (beta) connection weights per sublayer.
        # alpha -> softmax -> (hc,) mixing weights for the layer input.
        # beta  -> sigmoid -> (hc,) per-stream weights for distributing the output.
        self.alpha_attn = nn.Parameter(torch.full((hc,), -5.0))
        self.beta_attn  = nn.Parameter(torch.full((hc,), -5.0))
        self.alpha_moe  = nn.Parameter(torch.full((hc,), -5.0))
        self.beta_moe   = nn.Parameter(torch.full((hc,), -5.0))
        # Stream 0 starts dominant: close to standard residual.
        nn.init.constant_(self.alpha_attn[0], 5.0)
        nn.init.constant_(self.beta_attn[0],  5.0)
        nn.init.constant_(self.alpha_moe[0],  5.0)
        nn.init.constant_(self.beta_moe[0],   5.0)

    def _hc_input(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Weighted combination of streams -> single (B, T, C) input for a sublayer."""
        w = alpha.softmax(0)              # (hc,) — sums to 1
        return (h * w.view(1, 1, -1, 1)).sum(dim=2)  # (B, T, C)

    def _hc_update(
        self, h: torch.Tensor, out: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Distribute sublayer output back to all streams."""
        w = beta.sigmoid()                # (hc,) — each in (0,1)
        return h + out.unsqueeze(2) * w.view(1, 1, -1, 1)

    def forward(
        self, h: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Attention sublayer ---
        x_in   = self._hc_input(h, self.alpha_attn)
        attn_out = self.attn(self.rms_1(x_in), freqs_cis)
        h = self._hc_update(h, attn_out, self.beta_attn)

        # --- MoE sublayer ---
        x_in   = self._hc_input(h, self.alpha_moe)
        moe_out, lb_loss = self.moe(self.rms_2(x_in))
        h = self._hc_update(h, moe_out, self.beta_moe)

        return h, lb_loss


# ---------------------------------------------------------------------------
# 4. MTP — Multi-Token Prediction auxiliary head
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """Auxiliary head that predicts token t+2 (one step beyond the main head).

    Multi-Token Prediction (DeepSeek-V3, 2024) encourages the model to encode
    information about multiple future tokens in its hidden states, rather than
    just the immediate next token. This improves representations and can also be
    used for speculative decoding at inference.

    Implementation:
        A separate RMSNorm followed by the shared lm_head weight matrix.
        We reuse lm_head.weight (the vocabulary projection) since both heads
        project to the same vocabulary in the same embedding space.

    Loss contribution:
        mtp_loss = cross_entropy(mtp_logits[:, :-1, :], targets[:, 1:])
        (targets[:, 1:] are the t+2 tokens when main targets are t+1 tokens)
        Added to total loss with weight mtp_weight.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor, lm_weight: torch.Tensor) -> torch.Tensor:
        """Args:
            x:         (B, T, n_embd) final hidden states.
            lm_weight: (vocab_size, n_embd) — shared from lm_head.weight.
        Returns:
            logits: (B, T, vocab_size).
        """
        return F.linear(self.norm(x), lm_weight)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Full v2 transformer: MLA + MoE + Hyper-Connections + MTP.

    Forward pass:
        1. Token embedding: idx -> wte -> (B, T, n_embd).
        2. Init HC state: broadcast to (B, T, hc_expansion, n_embd).
        3. n_layer Blocks: each returns updated HC state + lb_loss.
        4. HC readout: learned weighted sum over streams -> (B, T, n_embd).
        5. Final RMSNorm -> lm_head -> logits.
        6. If targets: main LM loss + MTP loss + total load-balance loss.

    Checkpoint format (saved by train_v2.py):
        {"step", "model", "optimizer", "model_config", "train_config"}
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            rms_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # MTP auxiliary head (only if n_mtp > 0).
        if config.n_mtp > 0:
            self.mtp_head = MTPHead(config)

        # Learned readout weights: which HC stream(s) to read for the final output.
        # Init: heavily favor stream 0 (standard behavior).
        self.hc_readout = nn.Parameter(torch.full((config.hc_expansion,), -5.0))
        nn.init.constant_(self.hc_readout[0], 5.0)

        # Precomputed RoPE freqs for the rope portion of each head.
        freqs_cis = precompute_rope_freqs(
            rope_head_dim=config.rope_head_dim,
            seq_len=config.block_size,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """N(0, 0.02) for all Linear weights; scaled init for residual projections."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.block_size

        # Token embeddings only — RoPE encodes position inside MLA.
        x = self.transformer.wte(idx)          # (B, T, n_embd)

        # Initialize HC state: broadcast x across all streams.
        # Extra streams start as copies of x and diverge during training.
        hc = self.config.hc_expansion
        h = x.unsqueeze(2).expand(-1, -1, hc, -1).contiguous()  # (B, T, hc, n_embd)

        freqs_cis = self.freqs_cis[:T]
        lb_loss_total = torch.zeros(1, device=idx.device)
        for block in self.transformer.h:
            h, lb_loss = block(h, freqs_cis)
            lb_loss_total = lb_loss_total + lb_loss

        # HC readout: weighted sum of streams -> (B, T, n_embd).
        readout_w = self.hc_readout.softmax(0)                   # (hc,)
        x = (h * readout_w.view(1, 1, -1, 1)).sum(dim=2)         # (B, T, n_embd)
        x = self.transformer.rms_f(x)
        logits = self.lm_head(x)                                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Main next-token prediction loss.
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

            # MTP loss: predict token t+2 from position t.
            # targets[:, 1:] are the t+2 tokens (since targets[t] = token t+1).
            # We only compute over positions where a t+2 target exists (all but last).
            mtp_loss = torch.zeros(1, device=idx.device)
            if self.config.n_mtp > 0 and T > 1:
                mtp_logits = self.mtp_head(x, self.lm_head.weight)  # (B, T, vocab_size)
                mtp_loss = F.cross_entropy(
                    mtp_logits[:, :-1].reshape(-1, mtp_logits.size(-1)),
                    targets[:, 1:].reshape(-1),
                )

            loss = main_loss + self.config.mtp_weight * mtp_loss + lb_loss_total

        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device_type: str
    ) -> torch.optim.AdamW:
        """AdamW: weight decay on 2D+ params, none on 1D.

        1D params with no decay: RMSNorm weights, HC alpha/beta/readout scalars.
        2D+ params with decay: all Linear weight matrices, embedding table,
        MoE router, expert weights.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Same logic as v1; lb_loss ignored (no targets)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
