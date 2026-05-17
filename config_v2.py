"""Configuration for the v2 nanowhale-inspired architecture.

New fields vs config_v1.py:

  MLA:
    rope_head_dim / nope_head_dim  — per-head RoPE vs NoPE dimension split.
    q_lora_rank                    — Q latent bottleneck dimension.
    kv_lora_rank                   — shared KV latent bottleneck dimension.
    n_kv_head                      — number of KV heads before GQA expansion (1 = MQA).

  MoE:
    n_routed_experts               — total routed expert count.
    n_shared_experts               — always-active expert count.
    n_experts_per_tok              — top-k routing.
    moe_intermediate               — per-expert SwiGLU hidden dimension.
    router_scale                   — weight of the load-balance auxiliary loss.

  Hyper-Connections:
    hc_expansion                   — number of parallel hidden state streams.

  MTP:
    n_mtp                          — number of extra future tokens to predict (0 = off).
    mtp_weight                     — loss weight for the MTP auxiliary objective.

  Defaults are tuned for ~100M total parameters with the GPT-2 BPE tokenizer.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    """v2 model architecture: MLA + MoE + Hyper-Connections + MTP.

    Parameter breakdown at defaults (~106M total):
        Embeddings (wte, weight-tied):  50304 * 768 ≈  38.6M
        8 × MLA attention:              8 × ~1.0M   ≈   8.0M
        8 × MoE (4+1 experts, hid=640): 8 × ~7.4M   ≈  59.1M
        MTP head norm:                              ≈   0.6M
        Misc (norms, router, HC):                  ≈   0.2M
        Total                                      ≈ 106M
    """

    block_size: int = 2048
    vocab_size: int = 50304   # GPT-2 BPE, padded to multiple of 64

    # Depth. Fewer layers than v1 (12) because MoE multiplies effective capacity.
    n_layer: int = 8

    # --- MLA attention ---
    n_head:        int = 8    # query heads
    n_kv_head:     int = 1    # KV heads — 1 = MQA-style (nanowhale default)
    rope_head_dim: int = 32   # per-head dims that get RoPE rotation
    nope_head_dim: int = 64   # per-head dims that are position-free (NoPE)
                              # head_dim = rope_head_dim + nope_head_dim = 96
                              # n_head * head_dim must equal n_embd: 8 * 96 = 768 ✓
    n_embd:        int = 768  # must equal n_head * (rope_head_dim + nope_head_dim)
    q_lora_rank:   int = 192  # Q bottleneck dim. 0 would skip compression.
    kv_lora_rank:  int = 96   # KV shared latent dim. KV cache stores this + k_rope.

    # --- MoE feedforward ---
    n_routed_experts:  int   = 4      # total routed experts per layer
    n_shared_experts:  int   = 1      # always-active experts per layer
    n_experts_per_tok: int   = 2      # top-k routing (2 of 4 routed experts fire)
    moe_intermediate:  int   = 640    # per-expert SwiGLU hidden dim
    router_scale:      float = 1e-2   # load-balance loss weight (small — auxiliary)

    # --- Hyper-Connections ---
    hc_expansion: int = 2   # parallel hidden streams. nanowhale uses 4;
                            # we use 2 to keep memory comparable (larger n_embd).

    # --- Multi-Token Prediction ---
    n_mtp:      int   = 1    # extra future tokens to predict (0 = disabled)
    mtp_weight: float = 0.1  # weight of MTP loss in total loss

    # --- RoPE ---
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        head_dim = self.rope_head_dim + self.nope_head_dim
        assert self.n_embd == self.n_head * head_dim, (
            f"n_embd ({self.n_embd}) must equal n_head ({self.n_head}) "
            f"* head_dim ({head_dim})"
        )
        assert self.n_head % self.n_kv_head == 0, (
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        )
        assert self.n_experts_per_tok <= self.n_routed_experts


@dataclass
class TrainConfig:
    """Training hyperparameters for v2.

    Identical to v1 except log/ckpt dirs.
    MoE adds per-step compute (routing + expert dispatch), so wall-clock time
    per step will be ~30-50% longer than v1 on the same hardware.
    """

    total_batch_tokens: int = 524_288
    micro_batch_size:   int = 16
    seq_len:            int = 2048
    grad_accum_steps:   int = 16    # 16 * 2048 * 16 = 524288 ✓

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
    log_dir:        Path = Path("logs_v2")
    ckpt_dir:       Path = Path("checkpoints_v2")
    data_dir:       Path = Path("data/edu_fineweb10B")

    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    def __post_init__(self) -> None:
        assert self.tokens_per_step == self.total_batch_tokens
        assert self.min_lr <= self.max_lr
        assert self.warmup_steps < self.max_steps
