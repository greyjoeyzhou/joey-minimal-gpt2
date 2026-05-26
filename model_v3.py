"""v1 + MoE (Mixture of Experts) — the cleanest MoE upgrade.

This is model_v1.py with one change: the dense SwiGLU MLP in each Block is
replaced by a sparse MoE layer. Everything else — RMSNorm, RoPE, GQA — is
identical to v1.

Why MoE and not the other v2 changes?
    MoE is the highest-leverage architectural change for quality. MLA mainly
    helps inference (KV cache), Hyper-Connections are experimental, and MTP
    is a small auxiliary objective. MoE directly improves the quality/compute
    tradeoff by adding parameter capacity at no extra FLOPs per token.

The parameter/compute design:
    Naively keeping total params at ~114M (v1 level) while switching to MoE
    would *reduce* active compute per token (only 3/5 experts fire), making
    v3 weaker than v1 per token — an unfair comparison.

    The correct approach: **match active compute, not total params**.

    We want:  active_ffn_per_token ≈ v1_ffn_per_token
              (1 shared + 2 routed) × expert_size ≈ v1 SwiGLU size
              3 × expert_size ≈ 4.72M  →  intermediate ≈ 704

    This gives:
        v3 total params:         ~155M  (5 experts × 1.62M × 12 layers + rest)
        v3 active params/token:  ~116M  ≈ v1's 114M
        v3 "free extra" capacity: +39M  (non-active routed expert params)

    At the same FLOPs budget, v3 should beat v1 because the extra 39M params
    let each expert specialize on different token types / contexts.

MoE components:
    Expert:     single SwiGLU feedforward (same as v1 MLP but smaller).
    MoELayer:   1 shared expert (always fires) + n_routed experts (top-k
                selected per token by a learned router).
    Load balance loss: auxiliary loss that prevents router collapse (all
                tokens routing to the same 1-2 experts).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_v3 import GPTConfig


# ---------------------------------------------------------------------------
# RoPE helpers — identical to v1
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex RoPE phasors. See model_v1.py for full explanation."""
    assert head_dim % 2 == 0
    inv_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, inv_freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to Q or K. See model_v1.py for full explanation."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(x_complex * freqs_cis).flatten(3).to(x.dtype)


# ---------------------------------------------------------------------------
# Shared components — identical to v1
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Normalization. See model_v1.py for full explanation."""
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_inv = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms_inv * self.weight).to(x.dtype)


class CausalSelfAttentionGQA(nn.Module):
    """GQA attention with RoPE — identical to v1. See model_v1.py."""
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head
        self.n_rep     = config.n_head // config.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd,                    bias=False)
        self.o_proj.SCALE_INIT = 1  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


# ---------------------------------------------------------------------------
# MoE components — new in v3 (same logic as model_v2.py)
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single SwiGLU expert — same structure as v1's MLP, just smaller hidden dim.

    In v1: one expert, intermediate=2048, size=4.72M.
    In v3: five experts, intermediate=704, size=1.62M each.
    Active at once: 1 shared + 2 routed = 3 experts × 1.62M = 4.86M ≈ v1's 4.72M.
    """
    def __init__(self, n_embd: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.up_proj   = nn.Linear(n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, n_embd, bias=False)
        self.down_proj.SCALE_INIT = 1  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Sparse MoE feedforward layer.

    Design:
        n_shared_experts (=1): always process every token. This guarantees
            a baseline FFN signal regardless of routing decisions, and acts
            as a residual path that stabilizes early training.
        n_routed_experts (=4): compete per token. The router scores each
            candidate and the top-k are selected.

    Active FLOPs per token:
        (n_shared + n_experts_per_tok) × expert_forward_cost
        = (1 + 2) × 1.62M params × 2 FLOPs/param ≈ v1 dense MLP cost.

    Total parameters:
        (n_shared + n_routed) × expert_params
        = (1 + 4) × 1.62M = 8.10M per layer
        vs. v1's 4.72M per layer. The 3.38M "extra" per layer (40.6M over
        12 layers) is the free capacity that different experts can specialize.

    Router:
        A single Linear(n_embd, n_routed_experts) maps each token to expert
        logits. Softmax + top-k selects; selected weights are renormalized to
        sum to 1 before weighting outputs.

    Load balance loss:
        Prevents routing collapse (always picking the same 1-2 experts).
        L_aux = router_scale * n_experts * Σ_i(f_i * P_i)
        where f_i = fraction of tokens dispatched to expert i (discrete),
              P_i = mean routing probability for expert i (differentiable proxy).
        This loss is added to the main LM loss in GPT.forward().
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_routed  = config.n_routed_experts
        self.top_k     = config.n_experts_per_tok
        self.aux_scale = config.router_scale

        self.shared_experts = nn.ModuleList([
            Expert(config.n_embd, config.moe_intermediate)
            for _ in range(config.n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(config.n_embd, config.moe_intermediate)
            for _ in range(config.n_routed_experts)
        ])
        self.router = nn.Linear(config.n_embd, config.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        N = B * T
        flat_x = x.view(N, C)

        # Shared expert — always fires on the full batch.
        shared_out = sum(e(x) for e in self.shared_experts)

        # Router: softmax scores + top-k selection.
        scores = F.softmax(self.router(flat_x), dim=-1)       # (N, n_routed)
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # --- Sort-then-dispatch (replaces the old nested loop) ---
        #
        # Old approach: loop over (n_routed × top_k) = 8 iterations, each
        # launching a small kernel on ~N/5 scattered tokens.
        #
        # New approach:
        #   1. Flatten all N*k (token, expert) assignment pairs.
        #   2. Sort by expert ID → each expert's tokens become a contiguous slice.
        #   3. One kernel per expert on its slice → n_routed=4 kernel launches.
        #   4. Weight outputs and scatter-add back to original token positions.
        #
        # Why contiguous slices matter:
        #   - One larger matmul per expert (better tensor-core utilisation).
        #   - Sequential memory reads inside each chunk (coalesced access).
        #   - Average chunk size ~N*k/n_routed ≈ 2× larger than before.

        # Step 1: flatten the N*k (token, expert) pairs.
        token_ids  = torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)  # (N*k,)
        expert_ids = topk_idx.reshape(-1)   # (N*k,)
        weights    = topk_scores.reshape(-1)  # (N*k,)

        # Step 2: sort all pairs by expert ID.
        sort_order     = expert_ids.argsort()
        sorted_tokens  = token_ids[sort_order]   # (N*k,) — original token index
        sorted_weights = weights[sort_order]     # (N*k,)
        sorted_x       = flat_x[sorted_tokens]  # (N*k, C) — gathered inputs

        # One CPU-GPU sync to get per-expert token counts as Python ints.
        # This is unavoidable in pure PyTorch; a CUDA extension (e.g. Megablocks)
        # eliminates it by keeping counts on-device.
        counts = expert_ids[sort_order].bincount(minlength=self.n_routed).tolist()

        # Step 3: one kernel per expert on its contiguous chunk.
        routed_out = torch.empty_like(sorted_x)
        start = 0
        for expert, count in zip(self.routed_experts, counts):
            end = start + count
            if count > 0:
                routed_out[start:end] = expert(sorted_x[start:end])
            start = end

        # Step 4: weight by routing scores, then scatter-add back.
        # scatter_add_ accumulates contributions when a token is served by
        # multiple experts (top_k > 1).
        routed_out = routed_out * sorted_weights.unsqueeze(-1)
        out = torch.zeros(N, C, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, sorted_tokens.unsqueeze(-1).expand(-1, C), routed_out)

        lb_loss = self._load_balance_loss(scores, topk_idx)
        return shared_out + out.view(B, T, C), lb_loss

    def _load_balance_loss(
        self, scores: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        n = self.n_routed
        P = scores.mean(dim=(0, 1))                            # (n_routed,)
        one_hot = torch.zeros(*scores.shape, device=scores.device)
        one_hot.scatter_(-1, topk_idx, 1.0)
        f = one_hot.mean(dim=(0, 1)) / self.top_k              # (n_routed,)
        return self.aux_scale * n * (f * P).sum()


# ---------------------------------------------------------------------------
# Block — v1 structure, MoELayer in place of SwiGLU
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block: pre-RMSNorm -> GQA+RoPE -> residual -> pre-RMSNorm -> MoE -> residual.

    Identical to v1's Block except self.mlp (SwiGLU) is replaced by self.moe
    (MoELayer). The forward signature changes to return (x, lb_loss) so the
    GPT model can accumulate the load-balance loss over all layers.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn  = CausalSelfAttentionGQA(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.moe   = MoELayer(config)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.rms_1(x), freqs_cis)
        moe_out, lb_loss = self.moe(self.rms_2(x))
        x = x + moe_out
        return x, lb_loss


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """v3 GPT: v1 architecture (RMSNorm + RoPE + GQA) with MoE FFN.

    The only addition to v1's GPT is accumulating the per-layer MoE
    load-balance loss and adding it to the main LM loss.
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

        freqs_cis = precompute_rope_freqs(
            head_dim=config.n_embd // config.n_head,
            seq_len=config.block_size,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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

        x = self.transformer.wte(idx)   # (B, T, n_embd)
        freqs_cis = self.freqs_cis[:T]

        # Accumulate load-balance loss across all layers.
        lb_loss_total = torch.zeros(1, device=idx.device)
        for block in self.transformer.h:
            x, lb_loss = block(x, freqs_cis)
            lb_loss_total = lb_loss_total + lb_loss

        x = self.transformer.rms_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            # lb_loss_total is small (router_scale=1e-2) but ensures experts
            # share load. Without it, 1-2 experts dominate and the rest go unused.
            loss = main_loss + lb_loss_total

        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device_type: str
    ) -> torch.optim.AdamW:
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
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
