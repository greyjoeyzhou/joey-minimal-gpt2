"""Configuration dataclasses.

Two configs live here:

- GPTConfig: the model's architecture (layers, heads, embedding dim, ...).
  These rarely change once you've picked a model size.

- TrainConfig: the training loop's behavior (batch sizes, learning rates,
  intervals, paths, ...). These change per experiment.

Both are frozen-ish dataclasses with __post_init__ assertions so that bad
hyperparameter combinations crash *at startup* rather than 10 hours into a run.

Note: we deliberately do NOT use `from __future__ import annotations` here.
train.py introspects field types via `dataclasses.fields()` and compares them
to actual classes (e.g. `f.type is Path`). With future annotations, the types
would be stringified and the comparison would always fail.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPTConfig:
    """Architecture for a GPT-2-style decoder-only transformer.

    The defaults reproduce GPT-2 124M (the smallest model in the original paper).
    """

    # Maximum sequence length the model can attend over. GPT-2 used 1024.
    # The positional embedding matrix has this many rows.
    block_size: int = 1024

    # Vocabulary size. The GPT-2 BPE tokenizer has 50257 tokens, but we pad
    # to 50304 (= 50257 rounded up to a multiple of 64). The reason is purely
    # numerical: matrix multiplications on dimensions that are multiples of 64
    # use tensor cores more efficiently on modern GPUs. The extra rows are
    # never produced by the tokenizer, so they contribute zero loss/gradient.
    vocab_size: int = 50304

    # Depth (number of transformer blocks).
    n_layer: int = 12

    # Number of attention heads per block. Must divide n_embd evenly.
    n_head: int = 12

    # Hidden dimension (a.k.a. d_model). Each token is represented as a vector
    # of this size throughout the network.
    n_embd: int = 768

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )


@dataclass
class TrainConfig:
    """Training-loop hyperparameters.

    These mirror the build-nanogpt recipe. The defaults target a single
    RTX 5090. Adjust micro_batch_size + grad_accum_steps until you hit
    memory limits without overshooting tokens_per_step.
    """

    # --- Data / batch shape ---

    # The total number of tokens used in each optimizer step. GPT-2 used
    # ~0.5M tokens/step; we match that. Achieved via:
    #   tokens_per_step = micro_batch_size * seq_len * grad_accum_steps
    total_batch_tokens: int = 524_288  # 2^19

    # Per-forward-pass batch size. Set as high as VRAM allows.
    micro_batch_size: int = 32

    # Sequence length per sample. Must be <= GPTConfig.block_size.
    seq_len: int = 1024

    # Number of forward+backward passes accumulated before an optimizer step.
    # Larger grad_accum_steps -> smaller VRAM footprint but slower wall clock.
    grad_accum_steps: int = 16

    # --- Optimizer / LR schedule ---

    max_lr: float = 6e-4
    min_lr: float = 6e-5  # 10% of max_lr, per GPT-2 paper
    warmup_steps: int = 715  # linear warmup from 0 to max_lr
    max_steps: int = 19_073  # ~ 10B tokens / 524288 tokens per step
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # --- Eval / logging intervals (in optimizer steps) ---

    val_interval: int = 250  # how often to compute val loss
    val_iters: int = 20  # how many val batches to average over
    hella_interval: int = 1000  # how often to run HellaSwag
    save_interval: int = 5000  # how often to checkpoint
    log_dir: Path = Path("logs")
    ckpt_dir: Path = Path("checkpoints")
    data_dir: Path = Path("data/edu_fineweb10B")

    # --- Reproducibility ---

    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    def __post_init__(self) -> None:
        # The single most important consistency check: if the product doesn't
        # match total_batch_tokens, the schedule and loss curves won't match
        # the recipe, and the user won't realize until they've burned compute.
        assert self.tokens_per_step == self.total_batch_tokens, (
            f"micro_batch * seq_len * grad_accum = {self.tokens_per_step}, "
            f"but total_batch_tokens = {self.total_batch_tokens}. "
            "Adjust one of micro_batch_size/seq_len/grad_accum_steps."
        )
        assert self.min_lr <= self.max_lr
        assert self.warmup_steps < self.max_steps
