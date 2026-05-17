"""Configuration dataclasses for the v1 (modern) architecture.

Changes vs config.py:

  GPTConfig:
    - block_size: 1024 -> 2048 (RoPE generalizes better, and we want the
      longer context to actually be usable).
    - n_kv_head: new field for GQA key/value heads. Must divide n_head evenly.
      Setting n_kv_head == n_head reproduces standard MHA (no sharing).
    - rope_theta: RoPE base frequency. Larger values slow down the angular
      velocity, extending effective context. Llama 3 uses 500000 for very
      long context; 10000 is fine for 2048.
    - Removed: wpe is gone — position is encoded via RoPE, not a lookup table.

  TrainConfig:
    - seq_len: 1024 -> 2048 to match block_size.
    - micro_batch_size: 32 -> 16 to keep tokens_per_step = 524288.
      (16 * 2048 * 16 = 524288 = 2^19, same as before.)
    - log_dir / ckpt_dir: renamed to logs_v1 / checkpoints_v1 to avoid
      overwriting the GPT-2 run's logs.

Note: we deliberately do NOT use `from __future__ import annotations` here.
train_v1.py introspects field types via dataclasses.fields() to build the
argparser. With future annotations, types would be strings and comparisons
like `f.type is Path` would silently fail.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    """Architecture hyperparameters for the v1 modern transformer.

    The defaults reproduce a ~124M parameter model using modern components
    (RMSNorm, RoPE, SwiGLU, GQA). Parameter count is similar to GPT-2 124M
    because GQA saves KV projection params while SwiGLU's 2/3x hidden dim
    offsets the extra gate projection.
    """

    block_size: int   = 2048    # maximum sequence length
    vocab_size: int   = 50304   # GPT-2 BPE tokenizer, padded to multiple of 64
    n_layer:    int   = 12
    n_head:     int   = 12      # query attention heads
    n_kv_head:  int   = 4       # key/value heads for GQA (must divide n_head).
                                # n_head / n_kv_head = 3 query heads per KV group.
                                # Matches the Llama 3 8B ratio (32 Q / 8 KV = 4x).
    n_embd:     int   = 768
    rope_theta: float = 10000.0 # RoPE base. 10000 (original), 500000 (Llama 3 long ctx).

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        assert self.n_head % self.n_kv_head == 0, (
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        )


@dataclass
class TrainConfig:
    """Training-loop hyperparameters for the v1 model.

    All scheduler and optimizer settings are the same as config.py; only the
    batch shape and output paths differ.
    """

    # --- Data / batch shape ---

    total_batch_tokens: int = 524_288   # 2^19 — same as original
    micro_batch_size:   int = 16        # halved vs. original (seq_len doubled)
    seq_len:            int = 2048      # must be <= GPTConfig.block_size
    grad_accum_steps:   int = 16        # 16 * 2048 * 16 = 524288 ✓

    # --- Optimizer / LR schedule ---

    max_lr:       float = 6e-4
    min_lr:       float = 6e-5        # 10% of max_lr
    warmup_steps: int   = 715
    max_steps:    int   = 19_073      # ~10B tokens / 524288 tokens per step
    weight_decay: float = 0.1
    grad_clip:    float = 1.0

    # --- Eval / logging intervals (in optimizer steps) ---

    val_interval:   int  = 250
    val_iters:      int  = 20
    hella_interval: int  = 1000
    save_interval:  int  = 5000
    log_dir:        Path = Path("logs_v1")        # separate from logs/ (GPT-2 run)
    ckpt_dir:       Path = Path("checkpoints_v1")
    data_dir:       Path = Path("data/edu_fineweb10B")

    # --- Reproducibility ---

    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    def __post_init__(self) -> None:
        assert self.tokens_per_step == self.total_batch_tokens, (
            f"micro_batch * seq_len * grad_accum = {self.tokens_per_step}, "
            f"but total_batch_tokens = {self.total_batch_tokens}. "
            "Adjust micro_batch_size / seq_len / grad_accum_steps."
        )
        assert self.min_lr <= self.max_lr
        assert self.warmup_steps < self.max_steps
