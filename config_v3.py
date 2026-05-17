"""Configuration for v3: v1 architecture + MoE.

The critical design question is how to size the experts relative to v1's
dense MLP. There are two valid choices:

  ┌─────────────────────┬────────────┬───────────────────┬──────────────────────────┐
  │ Design              │ Total params│ Active per token  │ When to use              │
  ├─────────────────────┼────────────┼───────────────────┼──────────────────────────┤
  │ v1 (dense baseline) │  ~114M     │  ~114M            │ reference                │
  │ param-matched MoE   │  ~114M     │   ~68M            │ inference budget limited │
  │ compute-matched MoE │  ~155M     │  ~116M ≈ v1       │ fair quality comparison  │
  └─────────────────────┴────────────┴───────────────────┴──────────────────────────┘

  Param-matched: same total params as v1, but less active compute per token.
      v3 would be weaker than v1 at the same training-token budget.

  Compute-matched (default here): same active compute per token as v1, more
      total params. At the same FLOPs budget, v3 should beat v1 because:
        - 39M extra params across 12 layers let experts specialize.
        - Different token types can route to different experts.
      This is how Mixtral, DeepSeek, and all production MoE papers report results.

Expert sizing for compute-matched:
    v1 SwiGLU: intermediate = 2048, total MLP = 3 × 768 × 2048 = 4.72M/layer
    Active experts per token: 1 shared + 2 routed = 3
    Expert intermediate: 2048 / 3 ≈ 683 → rounded to 704 (nearest 64)
    Active MLP per token: 3 × (3 × 768 × 704) ≈ 4.87M ≈ v1's 4.72M  ✓
    Total MLP per layer:  5 × (3 × 768 × 704) ≈ 8.12M  (vs v1's 4.72M)
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    """v3 model: v1 (RMSNorm + RoPE + GQA + SwiGLU) with MoE FFN.

    Non-MoE fields are identical to v1.
    """

    block_size: int   = 2048
    vocab_size: int   = 50304
    n_layer:    int   = 12
    n_head:     int   = 12
    n_kv_head:  int   = 4       # GQA: 3 query heads per KV head
    n_embd:     int   = 768
    rope_theta: float = 10000.0

    # --- MoE (new vs v1) ---
    n_routed_experts:  int   = 4      # expert pool size
    n_shared_experts:  int   = 1      # always-active experts
    n_experts_per_tok: int   = 2      # top-k from routed pool
    # intermediate=704: compute-matched to v1's dense SwiGLU (see module docstring).
    # For param-matched instead, set intermediate=384.
    moe_intermediate:  int   = 704
    router_scale:      float = 1e-2   # load-balance loss weight

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        assert self.n_experts_per_tok <= self.n_routed_experts


@dataclass
class TrainConfig:
    """Same as v1 training config. Output dirs renamed to v3."""

    total_batch_tokens: int = 524_288
    micro_batch_size:   int = 16
    seq_len:            int = 2048
    grad_accum_steps:   int = 16

    max_lr:       float = 6e-4
    min_lr:       float = 6e-5
    warmup_steps: int   = 715
    max_steps:    int   = 19_073
    weight_decay: float = 0.1
    grad_clip:    float = 1.0

    val_interval:   int  = 250
    val_iters:      int  = 20
    hella_interval: int  = 1000
    save_interval:  int  = 5000
    log_dir:        Path = Path("logs_v3")
    ckpt_dir:       Path = Path("checkpoints_v3")
    data_dir:       Path = Path("data/edu_fineweb10B")

    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    def __post_init__(self) -> None:
        assert self.tokens_per_step == self.total_batch_tokens
        assert self.min_lr <= self.max_lr
        assert self.warmup_steps < self.max_steps
