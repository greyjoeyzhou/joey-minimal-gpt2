"""Modern decoder-only transformer (v1) — Llama-style GPT.

Upgrades from model.py (GPT-2):

  1. RMSNorm instead of LayerNorm — simpler, ~15% faster, same quality.
  2. RoPE (Rotary Position Embeddings) instead of learned absolute wpe.
  3. SwiGLU instead of GELU in the MLP — gated, more expressive per param.
  4. GQA (Grouped-Query Attention) instead of MHA — smaller KV cache.
  5. No biases in Linear layers — negligible quality impact, simpler code.
  6. Context window bumped to 2048 (block_size in config_v1.py).

Architecture comparison:

  | Subsystem       | model.py (GPT-2)            | model_v1.py (modern)           |
  |-----------------|-----------------------------|--------------------------------|
  | Normalization   | LayerNorm                   | RMSNorm                        |
  | Position enc.   | Learned abs. wpe lookup     | RoPE applied to Q and K        |
  | MLP activation  | GELU (tanh approx)          | SwiGLU (SiLU-gated)            |
  | MLP structure   | fc -> gelu -> proj (2 lin.) | gate+up -> silu*up -> down (3) |
  | Attn. heads     | MHA: n_kv_head == n_head    | GQA: n_kv_head < n_head        |
  | Linear biases   | Yes                         | No                             |
  | Context window  | 1024 tokens                 | 2048 tokens                    |

These four changes (RMSNorm, RoPE, SwiGLU, GQA) appear together in:
Llama 2/3, Mistral, Gemma, Falcon, and essentially all post-2023 open models.

References:
  RMSNorm  https://arxiv.org/abs/1910.07467
  RoPE     https://arxiv.org/abs/2104.09864
  SwiGLU   https://arxiv.org/abs/2002.05202
  GQA      https://arxiv.org/abs/2305.13245
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v1 import GPTConfig


# ---------------------------------------------------------------------------
# RoPE helpers (module-level, not nn.Module — just functions)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex RoPE frequency phasors.

    RoPE (Su et al. 2021) encodes absolute position as a *rotation* in the
    complex plane of Q and K vectors. The key property that makes it useful is
    that after rotation, the dot product q_m · k_n depends only on the
    *relative* position (m - n) — so the model naturally learns relative
    positional relationships, while the encoding is applied absolutely.

    For each position m and frequency index i:
        angle(m, i) = m / (theta ^ (2i / head_dim))

    We represent these as complex unit phasors:
        freqs_cis[m, i] = e^(j * angle(m, i)) = cos(angle) + j * sin(angle)

    The table is computed once at init, registered as a buffer, and sliced
    to actual T at forward time: freqs_cis[:T].

    Args:
        head_dim: Q/K head dimension. Must be even (we pair adjacent dims).
        seq_len:  Max sequence length to precompute for.
        theta:    RoPE base. Higher = slower freq progression = better for
                  long context. Original paper: 10000. Llama 3: 500000.

    Returns:
        Complex tensor of shape (seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # inv_freqs[i] = 1 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2).
    # Decreasing: low-index dimensions rotate fast (high frequency),
    # high-index dimensions rotate slow (low frequency).
    inv_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Positions 0 .. seq_len-1.
    t = torch.arange(seq_len, dtype=torch.float32)
    # angles[m, i] = t[m] * inv_freqs[i]. Shape: (seq_len, head_dim/2).
    angles = torch.outer(t, inv_freqs)
    # Complex phasor: cos(angle) + j*sin(angle). torch.polar(r, phi) = r*e^(j*phi).
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate Q or K by the precomputed RoPE phasors.

    We treat every consecutive pair of hidden dimensions as a 2D vector
    (a complex number) and multiply it by the corresponding phasor.
    Multiplication in the complex plane is a rotation, so:
        (d_0, d_1) -> rotated pair at angle m * inv_freqs[0]
        (d_2, d_3) -> rotated pair at angle m * inv_freqs[1]
        ...

    Args:
        x:         Float tensor of shape (B, n_head, T, head_dim).
        freqs_cis: Complex tensor of shape (T, head_dim // 2), from precompute_rope_freqs.

    Returns:
        Rotated tensor of the same shape and dtype as x.
    """
    # Reinterpret pairs of floats as complex numbers.
    # (B, n_head, T, head_dim) -> (B, n_head, T, head_dim/2) complex.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Broadcast freqs_cis: (T, head_dim/2) -> (1, 1, T, head_dim/2).
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    # Complex multiply = rotation. (a+jb)(c+jd) = (ac-bd) + j(ad+bc).
    x_rotated = x_complex * freqs_cis
    # Back to real: flatten last two dims (head_dim/2, 2) -> head_dim.
    return torch.view_as_real(x_rotated).flatten(3).to(x.dtype)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
    RMSNorm(x)  =  x          / sqrt(mean(x^2) + eps) * weight

    Why RMSNorm works as well as LayerNorm:
        The critical property LayerNorm provides is *scale invariance* — the
        output doesn't change if x is multiplied by a constant. That comes from
        the RMS denominator, not from mean-centering. The paper shows that
        removing mean-centering loses negligible quality while cutting ~15% of
        the normalization compute.

    The learnable `weight` (also called gain/scale) is initialized to 1, so
    RMSNorm starts as pure normalization; the model then learns per-channel
    scalings from there.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # shape: (dim,), init = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to fp32 for stable RMS. .rsqrt() = 1 / sqrt(...).
        rms_inv = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms_inv * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feedforward network (Shazeer, 2020 / Llama 2+).

    Vanilla GPT-2 MLP:
        x -> Linear(4d) -> GELU -> Linear(d)

    SwiGLU MLP:
        x -> gate_proj(x) -> SiLU  ⎤
                                    ⊗  -> down_proj -> output
        x -> up_proj(x)            ⎦

    The gate branch multiplies element-wise with the up branch. This makes
    each unit's activation conditional on the input: the gate can suppress
    or amplify any unit dynamically. Empirically this beats GELU with the
    same parameter count.

    Hidden dimension:
        Vanilla MLP: 4 * n_embd (2 projections).
        SwiGLU:      2/3 * 4 * n_embd (3 projections), rounded to multiple of 64.
        For n_embd=768: 2/3 * 4 * 768 = 2048.
        The 2/3 factor compensates for having 3 projections vs. 2, keeping the
        total FLOP count (and roughly the parameter count) comparable.

    SiLU (Sigmoid Linear Unit):
        silu(x) = x * sigmoid(x) = x / (1 + e^(-x))
        Also called "swish". Smooth, non-monotonic, empirically outperforms
        ReLU/GELU in the gated setting.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # 2/3 * 4 * n_embd rounded up to the nearest multiple of 64.
        hidden = int(2 / 3 * 4 * config.n_embd)
        hidden = ((hidden + 63) // 64) * 64
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
        # Residual projection: scaled init (std /= sqrt(2 * n_layer)).
        # Same trick as model.py — see GPT-2 paper section 2.3.
        self.down_proj.SCALE_INIT = 1  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(gate(x)) acts as a learned soft gate; up(x) carries the content.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# GQA attention
# ---------------------------------------------------------------------------

class CausalSelfAttentionGQA(nn.Module):
    """Grouped-Query Attention with RoPE (Ainslie et al., 2023).

    Standard MHA has n_head query heads AND n_head key/value head pairs.
    GQA uses n_head queries but only n_kv_head < n_head key/value pairs;
    each KV head is shared among (n_head // n_kv_head) query heads.

    Why GQA matters:
        During autoregressive generation, the model must store the past K
        and V tensors for every layer (the "KV cache"). With MHA, the cache
        grows as n_head * head_dim per token per layer. With GQA it grows as
        n_kv_head * head_dim — a factor of (n_head / n_kv_head) smaller.
        This allows significantly larger batches or longer sequences at the
        same VRAM budget. Quality relative to MHA is negligible in practice.

    Special cases:
        n_kv_head == n_head  => standard MHA (no sharing)
        n_kv_head == 1       => MQA (Multi-Query Attention, extreme sharing)

    Llama 3 8B:  n_head=32, n_kv_head=8  (4 queries per KV head)
    Our default: n_head=12, n_kv_head=4  (3 queries per KV head)

    Projections (all bias=False):
        q_proj: n_embd -> n_head    * head_dim
        k_proj: n_embd -> n_kv_head * head_dim
        v_proj: n_embd -> n_kv_head * head_dim
        o_proj: n_head * head_dim -> n_embd

    RoPE is applied to Q and K only (not V). V carries content, not position.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head
        self.n_rep     = config.n_head // config.n_kv_head  # query heads per KV head

        self.q_proj = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd,                    bias=False)
        self.o_proj.SCALE_INIT = 1  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project: Q is full n_head width; K, V are narrower (n_kv_head).
        q = self.q_proj(x)  # (B, T, n_head    * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)

        # Reshape to (B, n_*head, T, head_dim) for per-head operations.
        q = q.view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K. freqs_cis is (T, head_dim/2) complex.
        q = apply_rope(q, freqs_cis)  # (B, n_head,    T, head_dim)
        k = apply_rope(k, freqs_cis)  # (B, n_kv_head, T, head_dim)

        # Expand K and V to full n_head by repeating each KV head n_rep times.
        # repeat_interleave(n_rep, dim=1) repeats each head n_rep times in
        # sequence: [h0, h0, h0, h1, h1, h1, ...] for n_rep=3.
        # After this, K and V have shape (B, n_head, T, head_dim).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask.
        # PyTorch dispatches to flash-attention or mem-efficient kernel automatically.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassemble heads: (B, n_head, T, head_dim) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Block and GPT
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """One transformer layer: pre-RMSNorm -> GQA+RoPE -> residual -> pre-RMSNorm -> SwiGLU -> residual.

    freqs_cis is passed down from GPT.forward() so we only slice it once
    (at the top level) rather than once per block.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn  = CausalSelfAttentionGQA(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp   = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), freqs_cis)
        x = x + self.mlp(self.rms_2(x))
        return x


class GPT(nn.Module):
    """Full modern decoder-only transformer.

    Structure:
        wte:      token embeddings, (vocab_size, n_embd). No wpe.
        h:        n_layer transformer Blocks.
        rms_f:    final RMSNorm before the LM head.
        lm_head:  linear projection to logits, weight-tied with wte.
        freqs_cis: precomputed RoPE table (buffer, not a parameter).

    The absence of wpe (positional embedding lookup) is intentional: RoPE
    encodes position on-the-fly in Q and K, so no extra parameter table is
    needed. This also means the model can (in principle) generalize to lengths
    beyond block_size if we extend freqs_cis — though that requires other
    tricks (YaRN, etc.) in practice.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # No wpe here — position is encoded via RoPE in attention.
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            rms_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: lm_head and wte share the same weight tensor.
        # At 124M scale this saves ~38M params and gives a small quality win.
        # Note: Llama 3 and other large models typically untie these, since
        # the embedding table becomes a smaller fraction of total params.
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute the full RoPE frequency table for block_size positions.
        # Registered as a buffer: saved in state_dict, not trained, moves
        # to the right device with .to(device).
        freqs_cis = precompute_rope_freqs(
            head_dim=config.n_embd // config.n_head,
            seq_len=config.block_size,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Weight initialization following the GPT-2 paper's scheme.

        - Linear weights: N(0, 0.02). No biases (all Linear are bias=False).
        - Embedding:      N(0, 0.02).
        - RMSNorm weight: 1 (set by nn.Parameter(torch.ones(dim)) in RMSNorm).

        Scaled init for residual projections (o_proj, down_proj):
            std /= sqrt(2 * n_layer)
        Each transformer block contributes two residual branches (attn + mlp),
        so variance accumulates 2 * n_layer times along the residual stream.
        Dividing by sqrt(2 * n_layer) keeps the variance of the sum stable
        regardless of depth. See GPT-2 paper section 2.3.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # No bias init — all Linear layers have bias=False.
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx:     (B, T) token IDs.
            targets: (B, T) target IDs for LM loss, or None at inference.

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy if targets provided, else None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} > block_size {self.config.block_size}"
        )

        # Token embeddings only — no positional lookup.
        x = self.transformer.wte(idx)  # (B, T, n_embd)

        # Slice the precomputed RoPE table to the actual sequence length.
        # Shape: (T, head_dim/2) complex.
        freqs_cis = self.freqs_cis[:T]

        for block in self.transformer.h:
            x = block(x, freqs_cis)

        x = self.transformer.rms_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on 2D+ params, no decay on 1D params.

        With bias=False everywhere, the no-decay group only holds RMSNorm
        weight vectors (1D). All Linear weight matrices and the embedding
        table are 2D+ and get weight decay.

        Compared to model.py: the no-decay group is notably smaller because
        there are no bias vectors. This is fine — RMSNorm weights still
        benefit from no decay (they're scale factors, not weight matrices).
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
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling — identical logic to model.py.

        At each step the full sequence is fed through the model to get
        the next-token distribution. This is O(T^2) in sequence length.

        A production system would use a KV cache to avoid recomputing past
        keys and values. With GQA the KV cache is n_rep times smaller than
        MHA — that's one of the main practical benefits of GQA.

        Args:
            idx:            (B, T) starting context token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature:    > 1 = more random; < 1 = more greedy.
            top_k:          If set, sample from the top-k highest prob tokens.

        Returns:
            (B, T + max_new_tokens) token IDs.
        """
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
