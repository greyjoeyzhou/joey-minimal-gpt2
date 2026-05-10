"""GPT-2 model — decoder-only transformer with causal self-attention.

This file follows Karpathy's nanoGPT / build-nanogpt structure. It defines:

- CausalSelfAttention: multi-head self-attention with a causal mask.
- MLP: two-layer feedforward with GELU activation.
- Block: a single transformer layer (pre-LN, attention, residual, MLP, residual).
- GPT: the full model — token + position embeddings, N blocks, final LN, LM head.

Architecture choices (matching the original GPT-2 paper):

- Pre-LayerNorm (LN inside the residual stream's "left arm", not on the output).
- Tied input/output embeddings (lm_head.weight is the same tensor as wte.weight).
- Learned absolute position embeddings (not RoPE, not ALiBi — we follow GPT-2).
- GELU activation in the MLP (tanh approximation, matching GPT-2).
- Vocab size padded to a multiple of 64 for tensor-core efficiency.

Things we use that GPT-2 *didn't* have but are now standard:

- `F.scaled_dot_product_attention`: PyTorch's flash-attention-aware kernel.
- bf16 / `torch.compile` (handled in train.py, not here).
- Scaled init for the residual projections (GPT-2 paper's section 2.3 trick:
  divide std by sqrt(2 * n_layer)). This keeps activation norms stable as the
  network gets deeper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Causal means token at position t can attend to positions 0..t (not t+1..T).
    This is enforced by F.scaled_dot_product_attention with is_causal=True.

    Q, K, V are computed in a single fused linear layer (c_attn) for efficiency,
    then split. Output is projected by c_proj.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Fused projection: x -> (q, k, v) all at once. Saves a matmul vs three
        # separate projections, and lets PyTorch fuse the op.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection back to model dim.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Flag used in GPT._init_weights: c_proj is the "residual stream
        # projection" and its init std should be scaled by 1/sqrt(2*n_layer).
        # See _init_weights() and the GPT-2 paper, section 2.3.
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[attr-defined]

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project once, split into q/k/v. Each is (B, T, C).
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to multi-head: (B, T, n_head, head_dim) then transpose to
        # (B, n_head, T, head_dim) for the matmul.
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask. PyTorch picks the
        # best available kernel (flash, mem-efficient, math). On the 5090 with
        # bf16 inputs this dispatches to a flash-attention path.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # (B, n_head, T, head_dim) -> (B, T, C) for the next linear.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Position-wise feedforward: hidden -> 4*hidden -> hidden, with GELU."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # The classic transformer MLP expansion factor is 4. So d_model=768
        # gets blown up to 3072, then projected back.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GPT-2 used the tanh approximation of GELU. PyTorch's GELU(approximate='tanh')
        # matches that exactly. Modern best practice often uses exact GELU
        # (or SwiGLU), but we follow GPT-2.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # Same scaled-init flag as in attention.
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """A single transformer block: pre-LN -> Attention -> residual -> pre-LN -> MLP -> residual.

    Pre-LN (LayerNorm *before* the sublayer, not after) is what makes deep
    transformers trainable without learning rate warmup tricks. The original
    "Attention Is All You Need" used post-LN; GPT-2 onward use pre-LN.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: x flows through the residual stream unchanged; attn() and mlp()
        # *add* to it. This "residual stream" view is a useful mental model
        # for mechanistic interpretability.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full GPT-2 model.

    Structure:
        - wte:  token embeddings, (vocab_size, n_embd).
        - wpe:  learned positional embeddings, (block_size, n_embd).
        - h:    n_layer transformer Blocks.
        - ln_f: final LayerNorm applied before the LM head.
        - lm_head: linear layer projecting back to vocab logits, *weight-tied* to wte.

    Forward pass:
        token_ids -> embed + positional -> blocks -> final LN -> lm_head -> logits.
        If targets are passed, also compute cross-entropy loss.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # nn.ModuleDict so we can refer to wte/wpe/h/ln_f by name (matches the
        # HuggingFace GPT-2 naming, makes weight loading from HF straightforward).
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # LM head: projects final hidden state to vocab logits.
        # bias=False is the GPT-2 convention.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the same tensor backs both the token embedding lookup
        # and the LM head's output projection. This:
        #   1. Saves ~38M parameters at GPT-2 124M scale (vocab_size * n_embd).
        #   2. Encourages the model to use the same representation for "predicting
        #      this token" and "this token's input embedding", which the GPT-2
        #      paper reports as a small but consistent quality win.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights according to the GPT-2 paper's scheme.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights matching the GPT-2 paper.

        - Linear layers: normal(0, 0.02). Biases zero.
        - Embeddings: normal(0, 0.02).
        - LayerNorm: PyTorch defaults (weight=1, bias=0), already set.

        Scaled init for residual projections (c_proj inside attention and MLP):
        their std is divided by sqrt(2 * n_layer). Reason: each layer adds two
        residual contributions, and we want the variance of the residual stream
        to stay roughly constant with depth. See GPT-2 paper section 2.3.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: (B, T) token IDs.
            targets: (B, T) target token IDs, or None for inference.

        Returns:
            logits: (B, T, vocab_size).
            loss: scalar cross-entropy if targets provided, else None.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Position indices 0..T-1, shared across the batch.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        # Broadcast: pos_emb (T, n_embd) + tok_emb (B, T, n_embd) -> (B, T, n_embd).
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten (B, T, V) -> (B*T, V) and (B, T) -> (B*T,) for cross_entropy.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device_type: str
    ) -> torch.optim.AdamW:
        """Build the AdamW optimizer with two param groups.

        Decoupled weight decay (AdamW) should *not* be applied to:
          - 1D parameters: LayerNorm gamma/beta, biases.
          - Embedding tables (debatable, but standard practice excludes them).

        The convention here is the one nanoGPT uses: 2D+ params get weight decay,
        1D params don't. Since embeddings are 2D, they DO get decay in this
        scheme — matching karpathy's choice. (Some papers exclude embeddings;
        the difference is tiny in practice.)
        """
        # Collect all params that need gradients.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 2D+ get decay (Linear weight matrices and embeddings).
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        # 1D get no decay (biases, LayerNorm weights/biases).
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Fused AdamW: a single CUDA kernel for the whole update step. ~10%
        # speedup over the default. Only available on CUDA.
        use_fused = device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),  # GPT-2 paper values; beta2=0.95 (vs default 0.999)
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling.

        At each step:
          1. Run the model on the current (possibly cropped) sequence.
          2. Take the logits at the last position.
          3. Apply temperature, optional top-k truncation, softmax.
          4. Sample one token.
          5. Append.

        Args:
            idx: (B, T) starting context.
            max_new_tokens: number of new tokens to generate.
            temperature: > 1 = more random, < 1 = more greedy.
            top_k: if set, only sample from the top-k highest-probability tokens.

        Returns:
            (B, T + max_new_tokens) tensor of token IDs.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size. This is the simplest
            # approach; a more efficient one would use a KV cache.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            # Take logits at the last time step.
            logits = logits[:, -1, :] / temperature
            # Optional: zero out everything except top-k.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
