# Minimal GPT-2 Training Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, heavily commented GPT-2 124M training project that follows Karpathy's `build-nanogpt`, with workflow split between Mac (dev/smoke-test) and a single 5090 (data prep + training), so the user can learn modern transformer pre-training mechanics.

**Architecture:** Flat-layout Python project, raw PyTorch (no framework wrappers). The model and training loop are written as readable scripts with extensive inline comments. Data lives as on-disk `uint16` shards loaded via `np.memmap`. Logging is plain CSV. The codebase is structurally DDP-ready but wired for single-GPU only.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.5, tiktoken (GPT-2 BPE), HuggingFace `datasets`, numpy, pytest, ruff, uv (env/deps).

**Reference design spec:** `docs/superpowers/specs/2026-05-10-minimal-gpt2-design.md`

---

## Task 1: Project Skeleton

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `README.md` (placeholder; final version in Task 16)
- Create: empty directories `data/`, `logs/`, `checkpoints/` via `.gitkeep` only if needed (kept gitignored)

- [ ] **Step 1: Initialize git**

Run:
```bash
cd /Users/hang/Code/joey-minimal-gpt2
git init -b main
```
Expected: `Initialized empty Git repository`.

- [ ] **Step 2: Create `.gitignore`**

Write `.gitignore`:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
.python-version.local
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# uv
uv.lock

# Data / artifacts (large, not version-controlled)
data/
logs/
checkpoints/

# Editor / OS
.vscode/
.idea/
.DS_Store
*.swp

# Misc
*.log
*.bin
*.pt
```

> Note on `uv.lock`: most projects do commit `uv.lock`. For a learning project where the exact pin matters less than legibility, I'm ignoring it; revisit later if you want fully reproducible installs.

- [ ] **Step 3: Pin Python version**

Write `.python-version`:
```
3.12
```

- [ ] **Step 4: Create `pyproject.toml`**

Write `pyproject.toml`:
```toml
[project]
name = "minimal-gpt2"
version = "0.0.1"
description = "Minimal GPT-2 124M training project for learning."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.5",
    "tiktoken>=0.7",
    "datasets>=2.20",
    "numpy>=1.26",
    "requests>=2.31",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
# E,F = pyflakes/pycodestyle (defaults), I = isort, UP = pyupgrade, B = bugbear
select = ["E", "F", "I", "UP", "B"]
ignore = [
    "E501",  # line too long; we have long comment blocks intentionally
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 5: Create placeholder README**

Write `README.md`:
```markdown
# minimal-gpt2

A minimal GPT-2 124M training project, for learning. Follows Karpathy's `build-nanogpt`.

(README will be expanded in Task 16.)
```

- [ ] **Step 6: Install dependencies with uv**

Run:
```bash
uv sync --extra dev
```
Expected: creates `.venv/`, installs torch, tiktoken, datasets, etc. On Mac this installs CPU/MPS PyTorch.

> If `uv` isn't installed: `brew install uv`. The user uses `uv` for all Python projects (see CLAUDE.md).

- [ ] **Step 7: Verify the venv works**

Run:
```bash
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
Expected: prints torch version (e.g., `2.5.x`) and `True` on Apple Silicon.

- [ ] **Step 8: Commit**

```bash
git add .gitignore .python-version pyproject.toml README.md docs/superpowers/
git commit -m "chore: project skeleton with uv-managed deps; include design + plan docs"
```

> The `docs/superpowers/` directory contains the brainstorming spec
> (`specs/2026-05-10-minimal-gpt2-design.md`) and the implementation plan
> (`plans/2026-05-10-minimal-gpt2.md`). Both were written before this
> implementation; this is just the first opportunity to commit them.

---

## Task 2: `config.py` — Configuration Dataclasses

**Files:**
- Create: `config.py`
- Create: `tests/__init__.py` (empty, makes tests/ a package)
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:
```python
"""Tests for config dataclasses.

These are sanity checks — the configs are mostly data, but we want to verify
their __post_init__ assertions actually fire on bad inputs.
"""
import pytest

from config import GPTConfig, TrainConfig


def test_gpt_config_defaults_are_gpt2_124m():
    """The default GPTConfig should match GPT-2 124M architecture exactly."""
    cfg = GPTConfig()
    assert cfg.n_layer == 12
    assert cfg.n_head == 12
    assert cfg.n_embd == 768
    assert cfg.block_size == 1024
    # We pad vocab to a multiple of 64 for efficient matmul (50257 -> 50304).
    assert cfg.vocab_size == 50304


def test_train_config_token_budget_consistency():
    """Train config asserts micro_batch * seq_len * grad_accum == total_batch_tokens."""
    # Good config: 8 * 1024 * 64 = 524288
    cfg = TrainConfig(
        total_batch_tokens=524_288,
        micro_batch_size=8,
        seq_len=1024,
        grad_accum_steps=64,
    )
    assert cfg.tokens_per_step == 524_288

    # Bad config: product doesn't match total
    with pytest.raises(AssertionError):
        TrainConfig(
            total_batch_tokens=524_288,
            micro_batch_size=8,
            seq_len=1024,
            grad_accum_steps=63,  # off by one
        )
```

Also create empty `tests/__init__.py`:
```python
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_config.py -v
```
Expected: collection error (no `config` module).

- [ ] **Step 3: Implement `config.py`**

Create `config.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_config.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/__init__.py tests/test_config.py
git commit -m "feat: GPTConfig and TrainConfig dataclasses with sanity checks"
```

---

## Task 3: `model.py` — GPT-2 Model

This is the heart of the project. The code closely follows `build-nanogpt`'s `train_gpt2.py`, restructured into `model.py`-only contents with extensive inline comments.

**Files:**
- Create: `model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:
```python
"""Tests for the GPT model.

These run on Mac CPU in a few seconds. They check shapes, finite values, the
parameter count, and that generation is deterministic with a fixed seed.
"""
import torch

from config import GPTConfig
from model import GPT


def test_forward_pass_shapes():
    """Model should produce logits of shape (B, T, vocab_size) and a scalar loss."""
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128)
    model = GPT(cfg)
    B, T = 3, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(x, y)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_forward_without_targets_returns_none_loss():
    """When called without targets, loss should be None (inference path)."""
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_param_count_is_about_124m():
    """Default GPTConfig should yield a model with ~124M parameters.

    With weight tying (lm_head shares weights with wte), the actual count
    is around 124M. We allow a small tolerance because vocab padding adds
    ~36k extra embedding weights.
    """
    model = GPT(GPTConfig())
    n_params = sum(p.numel() for p in model.parameters())
    # GPT-2 124M is ~124,439,808 with tied weights and padded vocab.
    assert 123_000_000 < n_params < 126_000_000, f"got {n_params:,} params"


def test_weight_tying():
    """Token embedding and LM head should share the same tensor."""
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128))
    assert model.transformer.wte.weight is model.lm_head.weight


def test_generation_is_deterministic_with_seed():
    """Greedy generation with a fixed seed should be reproducible."""
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=128)
    model = GPT(cfg)
    model.eval()
    prompt = torch.tensor([[1, 2, 3]])

    torch.manual_seed(0)
    out1 = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=4)

    torch.manual_seed(0)
    out2 = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=4)

    assert torch.equal(out1, out2)
    assert out1.shape == (1, 8)  # 3 prompt + 5 generated
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_model.py -v
```
Expected: import error — `model` module not found.

- [ ] **Step 3: Implement `model.py`**

Create `model.py`:
```python
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

import math

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
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # 1D get no decay (biases, LayerNorm weights/biases).
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_model.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: GPT-2 model with attention, MLP, blocks, and generate()"
```

---

## Task 4: `utils.py` — LR Schedule, Device Detection, CSV Logger, Seeding

**Files:**
- Create: `utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_utils.py`:
```python
"""Tests for utils.py — lr schedule, device, csv logger, seeding."""
import csv
from pathlib import Path

import pytest
import torch

from utils import (
    CSVLogger,
    detect_device,
    get_lr,
    seed_everything,
)


def test_get_lr_warmup_phase():
    """During warmup, LR ramps linearly from 0 to max_lr."""
    max_lr = 6e-4
    warmup = 100
    max_steps = 1000
    min_lr = 6e-5
    # Step 0 (boundary): tiny but positive.
    assert get_lr(0, max_lr, min_lr, warmup, max_steps) == pytest.approx(max_lr / warmup)
    # Step warmup-1: very close to max_lr.
    assert get_lr(warmup - 1, max_lr, min_lr, warmup, max_steps) == pytest.approx(max_lr)


def test_get_lr_cosine_decay():
    """After warmup, LR cosine-decays from max_lr down to min_lr."""
    max_lr = 6e-4
    min_lr = 6e-5
    warmup = 100
    max_steps = 1000
    # Just after warmup: ~ max_lr.
    assert get_lr(warmup, max_lr, min_lr, warmup, max_steps) == pytest.approx(max_lr, rel=1e-3)
    # At the end: min_lr.
    assert get_lr(max_steps, max_lr, min_lr, warmup, max_steps) == pytest.approx(min_lr)
    # Past the end: stays at min_lr.
    assert get_lr(max_steps + 500, max_lr, min_lr, warmup, max_steps) == pytest.approx(min_lr)
    # Midpoint of cosine: between min_lr and max_lr.
    mid = get_lr((warmup + max_steps) // 2, max_lr, min_lr, warmup, max_steps)
    assert min_lr < mid < max_lr


def test_detect_device_returns_string():
    """detect_device should return one of 'cuda', 'mps', 'cpu'."""
    device = detect_device()
    assert device in ("cuda", "mps", "cpu")


def test_seed_everything_makes_torch_deterministic():
    """After seeding, two identical torch.randn calls should match."""
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_csv_logger_writes_rows(tmp_path: Path):
    """CSVLogger should create a file with the expected columns and rows."""
    log_path = tmp_path / "train.csv"
    logger = CSVLogger(log_path)
    logger.log_train(step=0, loss=4.5, lr=1e-5, dt_ms=300.0, tokens_per_sec=50_000.0, grad_norm=1.2)
    logger.log_val(step=100, val_loss=3.9)
    logger.log_hella(step=500, hella_acc=0.27)
    logger.close()

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["kind"] == "train"
    assert rows[0]["loss"] == "4.5"
    assert rows[1]["kind"] == "val"
    assert rows[1]["val_loss"] == "3.9"
    assert rows[2]["kind"] == "hella"
    assert rows[2]["hella_acc"] == "0.27"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_utils.py -v
```
Expected: import error — `utils` module not found.

- [ ] **Step 3: Implement `utils.py`**

Create `utils.py`:
```python
"""Utility helpers shared across training, eval, and sampling.

Kept intentionally tiny and dependency-free (numpy, torch, stdlib only).
"""
from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def detect_device() -> str:
    """Pick the best available device in order: cuda > mps > cpu.

    Returns the *device type string* ('cuda', 'mps', 'cpu') rather than a
    torch.device, because some torch APIs (autocast, AdamW fused) want the
    string form.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs.

    This makes runs reproducible. Note: full determinism on CUDA also requires
    setting `torch.use_deterministic_algorithms(True)` and CUBLAS env vars,
    which trade speed for exactness. We don't enable that — for training, we
    just want the *initial* state to be reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set PYTHONHASHSEED for dict ordering reproducibility in any subprocesses.
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Cosine LR schedule with linear warmup.

    Three phases:
      1. Steps 0 .. warmup_steps-1: linear ramp from ~0 to max_lr.
         (Specifically: lr = max_lr * (step + 1) / warmup_steps.)
      2. Steps warmup_steps .. max_steps: cosine decay from max_lr to min_lr.
      3. Steps > max_steps: stays at min_lr.

    The "+1" in warmup means step 0 still has a tiny but nonzero LR. This is
    intentional: the very first batch never gets a 0-learning-rate update.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay from max_lr -> min_lr over (max_steps - warmup_steps) steps.
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    # cos(0) = 1, cos(pi) = -1. coeff goes from 1 -> 0 over the decay.
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class CSVLogger:
    """Append-only CSV logger for training metrics.

    The CSV has a unified schema where each row is one event. The `kind`
    column tells you what kind of event it is — train, val, or hella. Only
    the columns relevant to that kind are filled (others are empty strings).

    This single-table format makes it trivial to load later:
        df = pd.read_csv("logs/train.csv")
        df[df["kind"] == "train"].plot(x="step", y="loss")
    """

    FIELDS = [
        "step",
        "kind",  # train / val / hella
        "loss",
        "val_loss",
        "hella_acc",
        "lr",
        "dt_ms",
        "tokens_per_sec",
        "grad_norm",
    ]

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # If the file doesn't exist, create it and write the header.
        write_header = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def _write_row(self, row: dict[str, Any]) -> None:
        # Fill missing fields with empty string for readability.
        full_row = {k: row.get(k, "") for k in self.FIELDS}
        self._writer.writerow(full_row)
        self._file.flush()  # tail -f and crash-resilience matter more than throughput

    def log_train(
        self,
        step: int,
        loss: float,
        lr: float,
        dt_ms: float,
        tokens_per_sec: float,
        grad_norm: float,
    ) -> None:
        self._write_row(
            dict(
                step=step,
                kind="train",
                loss=loss,
                lr=lr,
                dt_ms=dt_ms,
                tokens_per_sec=tokens_per_sec,
                grad_norm=grad_norm,
            )
        )

    def log_val(self, step: int, val_loss: float) -> None:
        self._write_row(dict(step=step, kind="val", val_loss=val_loss))

    def log_hella(self, step: int, hella_acc: float) -> None:
        self._write_row(dict(step=step, kind="hella", hella_acc=hella_acc))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CSVLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_utils.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: utils — lr schedule, device detection, CSV logger, seeding"
```

---

## Task 5: `scripts/prep_shakespeare.py` — Tiny Smoke-Test Dataset

**Files:**
- Create: `scripts/prep_shakespeare.py`

This script downloads tiny-shakespeare (~1MB), tokenizes it with GPT-2 BPE, and writes it as two `.bin` shards (one train, one val) under `data/shakespeare/`. Used for fast Mac smoke tests.

- [ ] **Step 1: Implement the script**

Create `scripts/prep_shakespeare.py`:
```python
"""Tiny Shakespeare dataset prep.

Downloads tinyshakespeare (~1 MB of plain text), tokenizes it with the GPT-2
BPE tokenizer, and writes train/val shards as raw uint16 .bin files.

Used by `tests/test_smoke_train.py` and for any local Mac smoke testing.
The full FineWeb-Edu pipeline is `scripts/prep_fineweb_edu.py`.

Run:
    uv run python scripts/prep_shakespeare.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import requests
import tiktoken

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_DIR = Path("data/shakespeare")
TRAIN_FRAC = 0.9


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUT_DIR / "input.txt"
    if not raw_path.exists():
        print(f"Downloading {URL}...")
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        raw_path.write_text(r.text)
    text = raw_path.read_text()
    print(f"Read {len(text):,} characters")

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    # encode_ordinary skips special-token handling — we just want raw BPE.
    tokens_arr = np.array(tokens, dtype=np.uint16)
    print(f"Tokenized to {len(tokens_arr):,} tokens (vocab size 50257)")

    split_idx = int(len(tokens_arr) * TRAIN_FRAC)
    train = tokens_arr[:split_idx]
    val = tokens_arr[split_idx:]

    train_path = OUT_DIR / "train.bin"
    val_path = OUT_DIR / "val.bin"
    train.tofile(train_path)
    val.tofile(val_path)
    print(f"Wrote {train_path} ({train.size:,} tokens)")
    print(f"Wrote {val_path}   ({val.size:,} tokens)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

Run:
```bash
uv run python scripts/prep_shakespeare.py
ls -la data/shakespeare/
```
Expected: `input.txt` (~1.1MB), `train.bin` (~600KB), `val.bin` (~60KB).

- [ ] **Step 3: Verify the bin files round-trip**

Run:
```bash
uv run python -c "
import numpy as np
import tiktoken
tokens = np.fromfile('data/shakespeare/train.bin', dtype=np.uint16)
enc = tiktoken.get_encoding('gpt2')
print('First 50 tokens decode to:', repr(enc.decode(tokens[:50].tolist())))
print('Total train tokens:', tokens.size)
"
```
Expected: the first 50 tokens decode to the opening line of the input text (`"First Citizen:\nBefore we proceed..."`).

- [ ] **Step 4: Commit**

```bash
git add scripts/prep_shakespeare.py
git commit -m "feat: tinyshakespeare data prep for Mac smoke tests"
```

---

## Task 6: `data.py` — Sharded Data Loader

**Files:**
- Create: `data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:
```python
"""Tests for the sharded data loader.

These rely on the tinyshakespeare data being prepared
(scripts/prep_shakespeare.py).
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from data import DataLoaderLite

SHAKE = Path("data/shakespeare")


def setup_module(module):
    """Skip the whole module if shakespeare data isn't prepared."""
    if not (SHAKE / "train.bin").exists():
        pytest.skip("Run scripts/prep_shakespeare.py first to generate test data.")


def test_loader_returns_correct_shapes():
    loader = DataLoaderLite(split="train", B=2, T=16, data_dir=SHAKE, shard_glob="*.bin")
    x, y = loader.next_batch()
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_loader_y_is_x_shifted_by_one():
    """For language modeling, y[i] should equal x[i+1] (the next-token target)."""
    loader = DataLoaderLite(split="train", B=1, T=32, data_dir=SHAKE, shard_glob="*.bin")
    x, y = loader.next_batch()
    # y[0, :-1] should equal x[0, 1:]
    assert torch.equal(y[0, :-1], x[0, 1:])


def test_loader_advances_position():
    """Consecutive next_batch() calls should return non-overlapping chunks."""
    loader = DataLoaderLite(split="train", B=1, T=16, data_dir=SHAKE, shard_glob="*.bin")
    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()
    assert not torch.equal(x1, x2)
    # In rank=0/world=1 setup, x2 should start where x1's targets ended.


def test_loader_rolls_over_to_next_shard():
    """When we exhaust a shard, the loader should roll forward (back to 0 if only one)."""
    # Use a tiny T so we definitely wrap.
    loader = DataLoaderLite(split="val", B=1, T=8, data_dir=SHAKE, shard_glob="*.bin")
    # The val shard has ~33k tokens. Read enough batches to exhaust it.
    val_tokens = np.fromfile(SHAKE / "val.bin", dtype=np.uint16).size
    n_batches = val_tokens // (1 * 8) + 5  # overshoot
    for _ in range(n_batches):
        x, y = loader.next_batch()
        assert x.shape == (1, 8)


def test_loader_respects_rank_world_size():
    """With world_size > 1, each rank should read a non-overlapping slice."""
    loader_a = DataLoaderLite(split="train", B=2, T=16, data_dir=SHAKE, shard_glob="*.bin", rank=0, world_size=2)
    loader_b = DataLoaderLite(split="train", B=2, T=16, data_dir=SHAKE, shard_glob="*.bin", rank=1, world_size=2)
    xa, _ = loader_a.next_batch()
    xb, _ = loader_b.next_batch()
    # Different ranks should get different data on the first call.
    assert not torch.equal(xa, xb)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_data.py -v
```
Expected: import error.

- [ ] **Step 3: Implement `data.py`**

Create `data.py`:
```python
"""On-disk sharded data loader for GPT pre-training.

Design:

- Each "shard" is a single .bin file of contiguous uint16 token IDs. No
  header, no compression — just np.fromfile or np.memmap to load.
- DataLoaderLite picks one shard at a time, hands out (x, y) batches from
  a sliding window, and rolls to the next shard when exhausted.
- For DDP, the loader is rank-aware: each call to next_batch advances by
  B*T*world_size, and rank `r` starts at offset r*B*T within each window.
  This way every rank sees a disjoint stripe of tokens.

This mirrors karpathy's DataLoaderLite in build-nanogpt almost exactly. Kept
"lite" because we deliberately skip prefetching, async loading, and shuffling.
For GPT-style training, sequential reads of large random-looking shards are
already cache-friendly enough.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _load_shard(path: Path) -> torch.Tensor:
    """Load a uint16 .bin shard into a torch.long tensor.

    We convert to long at load time (instead of every batch) because:
      - Embeddings expect int64 indices.
      - The conversion is cheap and happens once per shard.
      - Memory is fine: a 100M-token shard at int64 = 800 MB. We hold one at
        a time, and most shards won't be that big in dev settings. For very
        large shards you'd want memmap + per-batch cast — see the comments
        at the bottom of this file.
    """
    arr = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


class DataLoaderLite:
    """Stream batches of (x, y) from a directory of .bin token shards.

    Args:
        split: "train" or "val". Used to filter shard filenames.
        B: micro batch size.
        T: sequence length.
        data_dir: directory containing the shard files.
        shard_glob: glob pattern matching shards. E.g., "edufineweb_train_*.bin".
        rank: this process's rank in [0, world_size).
        world_size: total number of DDP processes.

    Conventions:
        - Shards are listed in sorted order.
        - For split filtering: shards whose name contains f"_{split}_" are picked.
          For tiny-shakespeare we also accept `{split}.bin` directly.
    """

    def __init__(
        self,
        split: str,
        B: int,
        T: int,
        data_dir: Path | str,
        shard_glob: str = "*.bin",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        assert split in {"train", "val"}
        assert 0 <= rank < world_size

        self.split = split
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size

        data_dir = Path(data_dir)
        all_shards = sorted(data_dir.glob(shard_glob))
        # Filter by split. Accept both "edufineweb_train_000001.bin" and "train.bin".
        shards = [
            s for s in all_shards
            if f"_{split}_" in s.name or s.stem == split
        ]
        if not shards:
            raise FileNotFoundError(
                f"No shards found for split='{split}' in {data_dir} "
                f"(glob={shard_glob!r}). Did you run the prep script?"
            )
        self.shards = shards
        self.reset()

    def reset(self) -> None:
        """Reset to the start of the first shard (start of an epoch)."""
        self.current_shard = 0
        self.tokens = _load_shard(self.shards[0])
        # rank `r` starts at offset r*B*T into the shard.
        self.current_position = self.B * self.T * self.rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (x, y) pair of shape (B, T).

        y is x shifted by one token, so cross_entropy(model(x), y) trains the
        model to predict each next token from the prefix.
        """
        B, T = self.B, self.T
        # We need B*T + 1 tokens (the +1 is so y can be shifted).
        end = self.current_position + B * T + 1
        buf = self.tokens[self.current_position : end]

        # If we overshot the shard, roll to the next one and retry from offset 0.
        if buf.size(0) < B * T + 1:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = _load_shard(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.rank
            end = self.current_position + B * T + 1
            buf = self.tokens[self.current_position : end]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance by B*T*world_size so adjacent ranks read non-overlapping stripes.
        self.current_position += B * T * self.world_size
        return x, y


# Notes for scaling up:
#
# 1. For very large shards (multi-GB), replace `_load_shard` with np.memmap
#    and slice + cast per batch. Tradeoff: per-batch overhead, but the OS
#    page cache handles random-ish reads well.
#
# 2. To do "true" epoch-aware training (shuffle shard order per epoch, etc.),
#    keep track of which shards you've seen and reset/permute when all are done.
#    For 10B-token / 19k-step runs, you won't even finish one epoch, so the
#    simple round-robin in reset() is fine.
#
# 3. For multi-machine DDP, you typically want the *same* shard list on every
#    rank, with the per-rank offset doing the actual splitting. That's what
#    we do here.
```

- [ ] **Step 4: Ensure shakespeare data exists, then run tests**

Run:
```bash
# (already done in Task 5, but re-run is idempotent)
uv run python scripts/prep_shakespeare.py
uv run pytest tests/test_data.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: DataLoaderLite — DDP-aware sharded token loader"
```

---

## Task 7: `train.py` — Training Loop & Smoke Test

**Files:**
- Create: `train.py`
- Create: `tests/test_smoke_train.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_smoke_train.py`:
```python
"""End-to-end smoke test of the training loop on tinyshakespeare.

Runs ~20 optimizer steps with a tiny model on the shakespeare shards. The
assertion is that the loss drops meaningfully. This is the canary that
catches *plumbing* bugs (wrong shapes, optimizer not stepping, loss not
flowing through autocast, etc.) before you push code to the 5090.

Runs in ~30 seconds on Mac MPS or CPU.
"""
from pathlib import Path

import pytest

from train import train_smoke


SHAKE = Path("data/shakespeare")


def test_smoke_train_loss_drops():
    """20 steps on shakespeare — final loss should be < 0.9 * initial."""
    if not (SHAKE / "train.bin").exists():
        pytest.skip("Run scripts/prep_shakespeare.py first.")
    losses = train_smoke(steps=20, micro_batch_size=4, seq_len=64, grad_accum_steps=1)
    assert losses[0] > losses[-1], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[-1] < losses[0] * 0.9, (
        f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
```

- [ ] **Step 2: Run the smoke test to confirm it fails**

Run:
```bash
uv run pytest tests/test_smoke_train.py -v
```
Expected: import error — `train` not found.

- [ ] **Step 3: Implement `train.py`**

Create `train.py`:
```python
"""GPT-2 124M training loop.

Two entry points:

- main(): the full training run. Reads CLI args, builds a TrainConfig, runs
  for cfg.max_steps. Writes CSV logs and checkpoints. Use this on the 5090.

- train_smoke(): a tiny in-process training run for the smoke test. Uses a
  toy model and the tinyshakespeare shards. Runs on Mac CPU/MPS in seconds.

The actual training step is identical in both paths; train_smoke just sets
small hyperparams and returns the loss curve for assertion.

Run on 5090:
    uv run python train.py
With overrides:
    uv run python train.py --micro_batch_size=16 --grad_accum_steps=32
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import fields
from pathlib import Path

import torch

from config import GPTConfig, TrainConfig
from data import DataLoaderLite
from model import GPT
from utils import CSVLogger, detect_device, get_lr, seed_everything


def _build_argparser() -> argparse.ArgumentParser:
    """CLI auto-derived from TrainConfig fields. Edit defaults in config.py."""
    parser = argparse.ArgumentParser(description="Train GPT-2 124M")
    defaults = TrainConfig()
    for f in fields(defaults):
        # Path types: parse as strings, cast manually below.
        if f.type is Path:
            parser.add_argument(f"--{f.name}", type=str, default=str(getattr(defaults, f.name)))
        elif f.type is int:
            parser.add_argument(f"--{f.name}", type=int, default=getattr(defaults, f.name))
        elif f.type is float:
            parser.add_argument(f"--{f.name}", type=float, default=getattr(defaults, f.name))
        else:
            parser.add_argument(f"--{f.name}", default=getattr(defaults, f.name))
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint to resume from. Empty = train from scratch.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile (default True; pass --no-compile to disable).",
    )
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    return parser


def _train_one_step(
    model: torch.nn.Module,
    loader: DataLoaderLite,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_accum_steps: int,
    grad_clip: float,
    lr: float,
) -> tuple[float, float]:
    """Execute one optimizer step (= grad_accum_steps micro-steps).

    Returns (loss_accum, grad_norm).
    """
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    # autocast dtype: bf16 on CUDA (no loss scaler needed), fp32 elsewhere.
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    for _ in range(grad_accum_steps):
        x, y = loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        # Scale loss so that backward() accumulates a *mean* gradient over the
        # grad_accum_steps micro-batches (not a sum).
        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Apply the LR set by the caller.
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()

    return loss_accum, float(grad_norm.item())


def main() -> None:
    args = _build_argparser().parse_args()

    # Build TrainConfig from CLI args. Manual Path casts where needed.
    cfg_kwargs = {f.name: getattr(args, f.name) for f in fields(TrainConfig())}
    for k in ("log_dir", "ckpt_dir", "data_dir"):
        cfg_kwargs[k] = Path(cfg_kwargs[k])
    cfg = TrainConfig(**cfg_kwargs)

    seed_everything(cfg.seed)
    device = detect_device()
    print(f"Device: {device}")
    print(f"Tokens per step: {cfg.tokens_per_step:,}")
    print(f"Max steps: {cfg.max_steps:,}")
    print(f"Total tokens to be trained: {cfg.tokens_per_step * cfg.max_steps:,}")

    # On CUDA, set float32 matmul precision to 'high' to allow TF32 on Ampere+.
    # On Blackwell (5090) this affects fp32 fallback paths; bf16 autocast is
    # the main precision regime.
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # --- Model ---
    model = GPT(GPTConfig()).to(device)
    if args.compile and device == "cuda":
        # torch.compile gives ~1.5-2x speedup on training. First step compiles,
        # so it'll appear slow; subsequent steps are fast.
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)  # type: ignore[assignment]

    # --- Optimizer ---
    # We access configure_optimizers via the original module even if compiled.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr, device_type=device
    )

    # --- Data ---
    train_loader = DataLoaderLite(
        split="train", B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir
    )
    val_loader = DataLoaderLite(
        split="val", B=cfg.micro_batch_size, T=cfg.seq_len, data_dir=cfg.data_dir
    )

    # --- Resume? ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    # --- Logger ---
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(cfg.log_dir / "train.csv")

    # --- Training loop ---
    print("Starting training loop")
    for step in range(start_step, cfg.max_steps):
        # Periodic eval (val loss).
        if step % cfg.val_interval == 0 and step > 0:
            val_loss = _evaluate(model, val_loader, cfg.val_iters, device)
            logger.log_val(step, val_loss)
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        # Periodic HellaSwag.
        if step % cfg.hella_interval == 0 and step > 0:
            from eval_hellaswag import evaluate_hellaswag  # lazy import

            acc = evaluate_hellaswag(raw_model, device)
            logger.log_hella(step, acc)
            print(f"step {step:6d} | hella_acc {acc:.4f}")

        # Periodic checkpoint.
        if step % cfg.save_interval == 0 and step > 0:
            ckpt_path = cfg.ckpt_dir / f"model_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"step {step:6d} | saved {ckpt_path}")

        # Training step.
        t0 = time.time()
        lr = get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        loss_accum, grad_norm = _train_one_step(
            model, train_loader, optimizer, device, cfg.grad_accum_steps, cfg.grad_clip, lr
        )
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = cfg.tokens_per_step / dt
        logger.log_train(
            step=step,
            loss=loss_accum,
            lr=lr,
            dt_ms=dt * 1000,
            tokens_per_sec=tokens_per_sec,
            grad_norm=grad_norm,
        )
        # Print every step early in training, then every 10 once we're settled.
        if step < 20 or step % 10 == 0:
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                f"dt {dt*1000:6.1f}ms | tok/s {tokens_per_sec:,.0f}"
            )

    logger.close()
    print("Training complete.")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, val_loader: DataLoaderLite, iters: int, device: str) -> float:
    """Mean cross-entropy over `iters` validation batches."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = val_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        total += float(loss.item())
    model.train()
    return total / iters


def train_smoke(
    steps: int = 20,
    micro_batch_size: int = 4,
    seq_len: int = 64,
    grad_accum_steps: int = 1,
) -> list[float]:
    """Tiny in-process training run on tinyshakespeare for the smoke test.

    Uses a *tiny* model (2 layers, 2 heads, n_embd=64) so it runs on a Mac
    CPU in seconds. Returns the per-step losses for assertion.
    """
    seed_everything(1337)
    device = detect_device()

    # Tiny model — not GPT-2 124M. We just want to verify the loop trains.
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=seq_len, vocab_size=50304)
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-3, device_type=device
    )

    loader = DataLoaderLite(
        split="train",
        B=micro_batch_size,
        T=seq_len,
        data_dir=Path("data/shakespeare"),
    )

    losses: list[float] = []
    for _ in range(steps):
        loss_accum, _ = _train_one_step(
            model, loader, optimizer, device, grad_accum_steps, grad_clip=1.0, lr=3e-3
        )
        losses.append(loss_accum)
    return losses


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test**

Run:
```bash
uv run pytest tests/test_smoke_train.py -v -s
```
Expected: 1 passed (loss drops from ~10.x to under 90% of that within 20 steps).

If loss doesn't drop enough: usually means the LR is too low for the tiny model. Bump `learning_rate=3e-3` in `train_smoke` to `1e-2` and re-run. If it diverges, lower it.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_smoke_train.py
git commit -m "feat: training loop with smoke test on tinyshakespeare"
```

---

## Task 8: `sample.py` — Text Generation From a Checkpoint

**Files:**
- Create: `sample.py`

- [ ] **Step 1: Implement `sample.py`**

Create `sample.py`:
```python
"""Generate text from a trained checkpoint.

Usage:
    uv run python sample.py --ckpt checkpoints/model_005000.pt \\
        --prompt "Hello, I'm a language model," \\
        --max_tokens 128 --n_samples 3 --temperature 0.8 --top_k 50

Notes:
    - At checkpoint creation time we save raw_model.state_dict() (not the
      torch.compile wrapper), so we can load it cleanly without referencing
      torch.compile internals.
    - GPT-2 BPE tokenization is via tiktoken("gpt2").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from config import GPTConfig
from model import GPT
from utils import detect_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = detect_device()

    # Load checkpoint.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # We saved `config` (TrainConfig) but not GPTConfig — the model arch is
    # the default 124M, so we use GPTConfig() here. If you ever vary arch,
    # also save GPTConfig and load it here.
    model = GPT(GPTConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Tokenize prompt.
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode_ordinary(args.prompt)
    # Repeat the prompt across n_samples so they all share context, but
    # different RNG draws yield different completions.
    x = torch.tensor([prompt_ids] * args.n_samples, dtype=torch.long, device=device)

    # Generate.
    out = model.generate(
        x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    )

    # Decode and print.
    for i in range(args.n_samples):
        text = enc.decode(out[i].tolist())
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke verify (optional, requires a checkpoint)**

If you have a smoke-test checkpoint already (e.g., from running the full smoke test in train.py mode), you can verify the CLI works. Otherwise this step is skipped — the real verification happens on the 5090 after training.

- [ ] **Step 3: Commit**

```bash
git add sample.py
git commit -m "feat: text generation from a checkpoint"
```

---

## Task 9: `scripts/prep_fineweb_edu.py` — Full FineWeb-Edu Tokenization

This script is intended to be run **on the 5090 box**, not the Mac, because it processes ~10B tokens (~50GB raw text -> ~20GB shards). It uses HuggingFace `datasets` to stream the data and a multiprocessing pool for tokenization.

**Files:**
- Create: `scripts/prep_fineweb_edu.py`

- [ ] **Step 1: Implement the script**

Create `scripts/prep_fineweb_edu.py`:
```python
"""Download and tokenize FineWeb-Edu 10B sample.

This is intended to be run on the 5090 box, not the Mac. It processes ~10B
tokens of educational web text and writes ~100 shards of 100M tokens each to
`data/edu_fineweb10B/`. Total disk footprint ~20 GB.

Architecture:
    - HuggingFace `datasets` streams the data in row-by-row.
    - A multiprocessing.Pool tokenizes documents in parallel (CPU-bound).
    - The main process accumulates tokens into a single buffer; when the
      buffer reaches `shard_size`, it's written to disk as a uint16 .bin.
    - Shard 0 is reserved for validation. Subsequent shards are training.

Run:
    uv run python scripts/prep_fineweb_edu.py
    # Optional: uv run python scripts/prep_fineweb_edu.py --shard_size 100_000_000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = Path("data/edu_fineweb10B")
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
SHARD_SIZE_DEFAULT = 100_000_000  # 100M tokens per shard

# tiktoken's GPT-2 BPE encoding. Special tokens:
#   <|endoftext|> = 50256 (EOT). We use it to separate documents.
_enc = tiktoken.get_encoding("gpt2")
_EOT = _enc._special_tokens["<|endoftext|>"]  # 50256


def _tokenize(doc: dict) -> np.ndarray:
    """Tokenize one doc (dict from HF), prepend EOT.

    Returning np.uint16 directly keeps the shard write cheap. We assert tokens
    fit in uint16, which is true for GPT-2 vocab (50257 < 65535).
    """
    text = doc["text"]
    tokens = [_EOT]
    tokens.extend(_enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.uint32)
    assert (tokens_np < 2**16).all(), "Token id out of uint16 range — vocab mismatch."
    return tokens_np.astype(np.uint16)


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    tokens.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE_DEFAULT)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stream the dataset. `split="train"` is HF's name for the whole sample.
    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG})...")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train")

    # Use ~half the CPU cores for tokenization to leave room for I/O and main.
    n_procs = max(1, (os.cpu_count() or 4) // 2)
    print(f"Tokenizing with {n_procs} workers, shard size = {args.shard_size:,} tokens")

    with mp.Pool(n_procs) as pool:
        shard_idx = 0
        # Preallocate the shard buffer once; copy into it.
        shard = np.empty(args.shard_size, dtype=np.uint16)
        pos = 0
        progress = tqdm(total=args.shard_size, unit="tok", desc=f"shard {shard_idx}")

        for tokens in pool.imap(_tokenize, ds, chunksize=16):
            # If this doc fits in the current shard, copy it in.
            if pos + len(tokens) < args.shard_size:
                shard[pos : pos + len(tokens)] = tokens
                pos += len(tokens)
                progress.update(len(tokens))
            else:
                # Flush the current shard. Take the prefix that fits, write,
                # then start a new shard with whatever's left.
                remaining = args.shard_size - pos
                shard[pos : pos + remaining] = tokens[:remaining]
                split = "val" if shard_idx == 0 else "train"
                out_path = args.out_dir / f"edufineweb_{split}_{shard_idx:06d}.bin"
                _write_shard(out_path, shard)
                progress.close()
                print(f"Wrote {out_path}")

                shard_idx += 1
                progress = tqdm(total=args.shard_size, unit="tok", desc=f"shard {shard_idx}")
                # Start the new shard with the leftover tokens.
                leftover = tokens[remaining:]
                shard[: len(leftover)] = leftover
                pos = len(leftover)
                progress.update(pos)

        # Flush any partial final shard (less than shard_size).
        if pos > 0:
            split = "val" if shard_idx == 0 else "train"
            out_path = args.out_dir / f"edufineweb_{split}_{shard_idx:06d}.bin"
            _write_shard(out_path, shard[:pos])
            progress.close()
            print(f"Wrote {out_path} (partial, {pos:,} tokens)")

    print(f"Done. {shard_idx + 1} shards written to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick check that imports work on Mac**

Run:
```bash
uv run python -c "import scripts.prep_fineweb_edu" 2>&1 | head -5
```
Expected: no error, or an `ImportError` about `datasets` which is fixed by `uv sync`. Don't run the script on Mac — it would consume ~50GB and many hours.

- [ ] **Step 3: Commit**

```bash
git add scripts/prep_fineweb_edu.py
git commit -m "feat: FineWeb-Edu 10B tokenization script"
```

---

## Task 10: `eval_hellaswag.py` — Zero-Shot HellaSwag

HellaSwag is a 4-way multiple-choice commonsense benchmark. For each example we have a context and 4 candidate endings; the model must pick the most likely one. We score each ending by per-token NLL of the ending given the context, and pick the lowest-NLL one. GPT-2 124M typically scores ~28-30%.

**Files:**
- Create: `eval_hellaswag.py`

- [ ] **Step 1: Implement `eval_hellaswag.py`**

Create `eval_hellaswag.py`:
```python
"""Zero-shot HellaSwag evaluation.

HellaSwag (Zellers et al. 2019) is a multi-choice commonsense benchmark.
Each example has:
  - ctx: a context sentence.
  - endings: 4 candidate continuations.
  - label: the correct continuation index.

Scoring:
    For each (ctx, ending) pair, we form ctx + ending, run it through the
    model, and compute the average per-token cross-entropy on the *ending
    tokens only* (we don't include context loss). Lowest NLL wins.

This is the standard "completion" zero-shot eval. GPT-2 124M baseline ~28-29%;
random is 25%.

Usage as a module:
    from eval_hellaswag import evaluate_hellaswag
    acc = evaluate_hellaswag(model, device, max_examples=1000)

Standalone:
    uv run python eval_hellaswag.py --ckpt checkpoints/model_005000.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import tiktoken
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import GPTConfig
from model import GPT
from utils import detect_device

URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
LOCAL = Path("data/hellaswag/hellaswag_val.jsonl")


def _download_if_needed() -> Path:
    if LOCAL.exists():
        return LOCAL
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    LOCAL.write_bytes(r.content)
    return LOCAL


def _render_example(example: dict, enc) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Render one example as (tokens, mask, label).

    - tokens: shape (4, T) — context+ending for each of 4 choices, right-padded.
    - mask:   shape (4, T) — 1 where ending tokens are, 0 for context or pad.
    - label:  int — correct ending index.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_ids = enc.encode_ordinary(ctx)
    rows = []
    masks = []
    for ending in endings:
        # Endings begin with a leading space in HellaSwag. Encode as part of
        # the same string flow so BPE works correctly.
        end_ids = enc.encode_ordinary(" " + ending)
        row = ctx_ids + end_ids
        mask = [0] * len(ctx_ids) + [1] * len(end_ids)
        rows.append(row)
        masks.append(mask)

    # Pad to the max length in this 4-way set with EOT (50256). Mask stays 0
    # on padding so they don't contribute to the NLL.
    max_len = max(len(r) for r in rows)
    PAD = 50256
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(r)] = torch.tensor(r)
        tokens[i, len(r) :] = PAD
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, label


@torch.no_grad()
def evaluate_hellaswag(
    model: torch.nn.Module,
    device: str,
    max_examples: int | None = None,
) -> float:
    """Run HellaSwag on the model. Returns accuracy in [0, 1]."""
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    path = _download_if_needed()

    correct = 0
    total = 0
    with open(path) as f:
        for line_idx, line in enumerate(tqdm(f, desc="hellaswag")):
            if max_examples is not None and line_idx >= max_examples:
                break
            example = json.loads(line)
            tokens, mask, label = _render_example(example, enc)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # Forward pass on all 4 candidates as one batch.
            autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
                logits, _ = model(tokens)
            # Shift: logit at position i predicts token at position i+1.
            shift_logits = logits[..., :-1, :].contiguous()  # (4, T-1, V)
            shift_targets = tokens[..., 1:].contiguous()  # (4, T-1)
            shift_mask = mask[..., 1:].contiguous().float()  # (4, T-1)

            # Per-token loss (no reduction).
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(
                shift_targets.size()
            )
            # Masked sum / masked count = mean loss on ending tokens only.
            sum_loss = (loss * shift_mask).sum(dim=1)
            n_tokens = shift_mask.sum(dim=1).clamp(min=1)
            avg_loss = sum_loss / n_tokens  # (4,)

            pred = int(avg_loss.argmin().item())
            if pred == label:
                correct += 1
            total += 1

    model.train()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    device = detect_device()
    model = GPT(GPTConfig()).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    acc = evaluate_hellaswag(model, device, max_examples=args.max_examples)
    print(f"HellaSwag accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file imports correctly**

Run:
```bash
uv run python -c "from eval_hellaswag import evaluate_hellaswag; print('ok')"
```
Expected: `ok`.

(Skip running the actual eval on Mac without a checkpoint — it's documented for the 5090.)

- [ ] **Step 3: Commit**

```bash
git add eval_hellaswag.py
git commit -m "feat: zero-shot HellaSwag eval"
```

---

## Task 11: `docs/01-architecture.md`

**Files:**
- Create: `docs/01-architecture.md`

- [ ] **Step 1: Write the doc**

Create `docs/01-architecture.md`. The doc should cover:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/01-architecture.md
git commit -m "docs: GPT-2 architecture walkthrough"
```

---

## Task 12: `docs/02-data-pipeline.md`

**Files:**
- Create: `docs/02-data-pipeline.md`

- [ ] **Step 1: Write the doc**

Create `docs/02-data-pipeline.md`. The doc should cover:

```markdown
# Data Pipeline: From FineWeb-Edu to Token Shards

This document explains how raw text becomes the (x, y) tensors the training
loop sees.

## The dataset: FineWeb-Edu (10B sample)

FineWeb-Edu is HuggingFace's curated subset of Common Crawl filtered for
educational content. It's the same dataset Karpathy's "Reproducing GPT-2"
video uses. The `sample-10BT` configuration is ~10B tokens (~50 GB raw text).

Why this and not OpenWebText / The Pile / C4?

- Higher quality on average (the "edu" filter removes a lot of low-signal web
  spam).
- Modern (2024), so it reflects current best practice.
- Available as a clean HuggingFace `datasets` stream — no manual reddit/web
  scraping.

## Tokenization: GPT-2 BPE (via tiktoken)

We use `tiktoken.get_encoding("gpt2")`. This is the same 50257-token BPE
vocabulary the original GPT-2 paper used. Same tokenizer means our token
counts and loss curves are directly comparable to GPT-2 in literature.

Each document is tokenized independently, then concatenated with an
`<|endoftext|>` (id 50256) separator. This signals "the previous document
ended" and lets the model learn document boundaries.

We pad the vocab from 50257 to 50304 in the *model* (not the tokenizer) for
matmul efficiency on tensor cores. The extra 47 rows in the embedding table
are never produced by tiktoken, so they contribute zero gradient.

## Sharding: uint16 .bin files

`scripts/prep_fineweb_edu.py` writes shards of 100M tokens each, as raw
`uint16` arrays on disk. Naming:

    data/edu_fineweb10B/
        edufineweb_val_000000.bin     # the first 100M tokens, reserved for val
        edufineweb_train_000001.bin
        edufineweb_train_000002.bin
        ...
        edufineweb_train_000099.bin   # ~last shard

Why uint16?

- 50304 < 65536, so it fits.
- Halves disk usage vs uint32.
- No header, no metadata. The format is "interpret this file as an array of
  uint16s." Loaders use `np.fromfile` or `np.memmap`.

Total disk: ~10B tokens × 2 bytes = ~20 GB.

Why not parquet / arrow / safetensors?

- Speed: `np.fromfile` of a flat uint16 array is the fastest possible read
  path. No deserialization overhead.
- Simplicity: you can `xxd` a shard and immediately understand the bytes.
- Resumability: shards are independent; if the prep script crashes, you
  re-run it (HF's cache is resumable, the script writes shard-by-shard).

## The loader: `data.py::DataLoaderLite`

Given a shard, the loader produces (x, y) windows of shape (B, T) where:

- `x[i, t]` = token at position t.
- `y[i, t]` = the next token, i.e., token at position t+1.

So `loss = cross_entropy(model(x), y)` trains the model to predict each next
token from the prefix.

### Window advancement

Each `next_batch()` advances by `B * T * world_size`. With `world_size=1`,
batches are contiguous. With DDP (`world_size > 1`), rank `r` starts at
offset `r*B*T` and skips ahead in lockstep — so the union of ranks reads a
disjoint stripe through each shard.

### Shard rollover

When a shard is exhausted, the loader rolls to the next shard. This is a
"round-robin within a split" loop, not a true epoch — for a 10B-token,
19k-step run we won't see the entire dataset, so per-epoch shuffling
doesn't matter.

## Working with the prep scripts

### Mac (development)

Use `scripts/prep_shakespeare.py` only. It downloads ~1 MB of tinyshakespeare
and writes `data/shakespeare/{train,val}.bin`. This is enough to smoke-test
the entire training loop.

### 5090 workstation (real training)

Run `scripts/prep_fineweb_edu.py` once. Expect:
- A few hours of wall time (CPU tokenization is the bottleneck).
- ~50 GB transient download from HF (cached at `~/.cache/huggingface/`).
- ~20 GB persistent on disk under `data/edu_fineweb10B/`.

You can blow away `~/.cache/huggingface/` after prep is done if you need disk
back.

## Verifying a shard

Roundtrip:

```python
import numpy as np, tiktoken
toks = np.fromfile("data/edu_fineweb10B/edufineweb_train_000001.bin", dtype=np.uint16)
enc = tiktoken.get_encoding("gpt2")
print(enc.decode(toks[:200].tolist()))
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/02-data-pipeline.md
git commit -m "docs: data pipeline walkthrough"
```

---

## Task 13: `docs/03-training-recipe.md`

**Files:**
- Create: `docs/03-training-recipe.md`

- [ ] **Step 1: Write the doc**

Create `docs/03-training-recipe.md`:

```markdown
# Training Recipe

This is the "what knobs and why" for `train.py`.

## Optimizer: AdamW

```
AdamW(params, lr=schedule(step), betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
```

- `betas=(0.9, 0.95)`: GPT-2 paper. The default torch AdamW uses 0.999 for
  beta2; 0.95 makes the second-moment estimate more responsive, which seems
  to matter for language modeling.
- `weight_decay=0.1`: applied only to 2D+ params (Linear weights, embeddings).
  1D params (biases, LN params) get no decay. This is *decoupled* weight
  decay (the W in AdamW): the decay is applied separately from the gradient
  step, not folded into the gradient.
- `fused=True` (CUDA only): one CUDA kernel for the whole optimizer step.

## Learning rate schedule

```
LR(step):
  if step < warmup_steps:          lr = max_lr * (step+1) / warmup_steps
  elif step >= max_steps:          lr = min_lr
  else:                            lr = min_lr + 0.5 * (1 + cos(pi * decay_ratio)) * (max_lr - min_lr)
```

Three phases: linear warmup, cosine decay, flat at min_lr.

Numbers:
- max_lr = 6e-4
- min_lr = 6e-5 (10% of max)
- warmup_steps = 715 (~ 375M tokens of warmup, matching GPT-3 paper at this scale)
- max_steps = 19_073 (~ 10B tokens at 524288 tokens/step)

Why cosine? Empirically it's a slightly smoother LR ramp-down than linear and
gives a small quality boost. Why not constant-then-decay? Cosine has been the
default for transformer pretraining since GPT-2.

## Token budget per step

The "magic number" is 524288 tokens per optimizer step. This is 2^19, the
batch the GPT-2 paper used. We hit it via gradient accumulation:

    tokens_per_step = micro_batch_size × seq_len × grad_accum_steps

On a single 5090, expect `seq_len=1024` and you'll tune `micro_batch_size`
and `grad_accum_steps` to multiply to 524288.

Starting guess: B=32, T=1024, grad_accum=16. If OOM, B=16/accum=32, etc.

## Gradient accumulation

Instead of running one giant batch, we run `grad_accum_steps` smaller batches
and accumulate gradients before stepping the optimizer:

```python
optimizer.zero_grad()
for _ in range(grad_accum_steps):
    x, y = loader.next_batch()
    loss = model(x, y)
    (loss / grad_accum_steps).backward()  # accumulate scaled gradients
optimizer.step()
```

The division by `grad_accum_steps` is so that `backward()` accumulates a
*mean* gradient (not a sum). The gradient that gets passed to optimizer.step()
is mathematically equivalent to one big batch of size `B * grad_accum_steps`.

## Mixed precision (bfloat16)

We wrap the forward pass in `torch.autocast(dtype=torch.bfloat16)`:

- Activations and intermediate computations are bf16.
- Optimizer states (Adam m, v) remain fp32.
- Gradients accumulated in fp32.

bf16 (vs fp16) has the same exponent range as fp32 (8 bits), so we don't need
loss scaling. The 5090 (Blackwell) has fast bf16 tensor cores.

Memory and speed math:
- bf16 activations halve the activation memory vs fp32.
- Tensor cores at bf16 are ~2x throughput vs fp32 on Blackwell.
- Net: ~1.5-2x speedup, half the activation memory.

## `torch.compile`

We wrap the model: `model = torch.compile(model)`. PyTorch's TorchInductor
JIT-compiles the model graph into fused CUDA kernels. First step is slow
(compiling); subsequent steps are 1.5-2x faster than eager mode.

We access the original (uncompiled) module via `model._orig_mod` whenever we
need to call non-forward methods like `configure_optimizers`.

## Gradient clipping

```
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Computes the global L2 norm of all gradients; if it exceeds 1.0, scales
everything down proportionally. Prevents single-step gradient spikes from
destabilizing training.

We log the grad norm at every step — sudden jumps are an early warning of
divergence.

## What we don't tune (intentionally)

Just to repeat the design's "out of scope" list with the recipe context:

- No DDP / FSDP / ZeRO — model fits on one 5090.
- No LR finder / hyperparam search — we use GPT-2's published recipe.
- No EMA, no SWA, no model averaging.
- No grad accumulation across multiple optimizer "passes" (more than 1
  optimizer step per gradient batch).
- No 8-bit AdamW / no optimizer offloading — fits in VRAM.

## Reading the training output

Each row in `logs/train.csv` is one of:

- `kind=train`: per-step loss, lr, dt, tokens/sec, grad norm.
- `kind=val`: averaged val loss at a `val_interval` boundary.
- `kind=hella`: HellaSwag accuracy at a `hella_interval` boundary.

To plot:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("logs/train.csv")
train = df[df["kind"] == "train"]
plt.plot(train["step"], train["loss"]); plt.yscale("log")
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/03-training-recipe.md
git commit -m "docs: training recipe walkthrough"
```

---

## Task 14: `docs/04-hardware-5090.md`

**Files:**
- Create: `docs/04-hardware-5090.md`

- [ ] **Step 1: Write the doc**

Create `docs/04-hardware-5090.md`:

```markdown
# Hardware: Training on a Single RTX 5090

This is the "how do I actually run it" companion to the training recipe.

## The 5090 spec sheet (relevant bits)

- 32 GB GDDR7 VRAM
- ~1.8 PFLOPS bf16/fp16 tensor cores
- ~3.4 PFLOPS fp4 (we don't use)
- Blackwell architecture: native bf16, full SDPA / flash attention support.

## Memory budget at 124M model

Rough numbers for `micro_batch=32, seq_len=1024`:

| Component | Size |
|---|---|
| Model params (bf16 + fp32 master copy) | ~750 MB |
| AdamW state (m, v in fp32) | ~1 GB |
| Activations (12 layers × B × T × n_embd × ~5 buffers × 2 bytes) | ~6-10 GB |
| KV / scratch buffers | ~1 GB |
| **Total** | **~10-12 GB** |

That leaves ~20 GB of headroom on 32 GB. You can comfortably push
`micro_batch_size` to 64 or even 128 before activations dominate.

## Tuning playbook

1. Run prep first: `uv run python scripts/prep_fineweb_edu.py`. Verify ~100
   shards in `data/edu_fineweb10B/`.
2. Run a smoke training pass — *very few steps*, default config:
   ```bash
   uv run python train.py --max_steps 50 --val_interval 100 --hella_interval 100 --save_interval 100
   ```
3. Watch `nvidia-smi` in another shell. Note GPU memory usage at steady state.
4. If memory is < 25 GB: double `micro_batch_size`, halve `grad_accum_steps`
   so `tokens_per_step` stays at 524288. Restart.
5. If you OOM: halve `micro_batch_size`, double `grad_accum_steps`. Restart.
6. Aim for the highest `micro_batch_size` that fits without OOM, leaving
   ~10% memory headroom (so `torch.compile`'s graph recompiles for varied
   shapes don't push you over).
7. Note the throughput (`tokens_per_sec` in train.csv). Reasonable target:
   50-80k tok/s. If you're at 20k, something is wrong.

## Throughput sanity checks

If `tokens_per_sec` is much lower than expected:

- `torch.compile` is disabled (check args, expect ~1.5x penalty without it).
- bf16 autocast is disabled (check `device=='cuda'` in train.py, expect ~2x).
- Data loader is the bottleneck (rare with mmap; would show up as low GPU
  utilization).
- Logging too frequently (we flush CSV every step, but that's tiny).

## Full run wall clock estimate

At 524288 tokens/step × 19073 steps = 10B tokens.

| tokens/sec | wall time |
|---|---|
| 50,000 | ~55 hours |
| 70,000 | ~40 hours |
| 100,000 | ~28 hours |

So expect a roughly 1.5-2 day run, end to end. Worth doing a 100-step
canary run first to project the actual rate.

## Resume after interruption

```bash
uv run python train.py --resume checkpoints/model_005000.pt
```

`save_interval=5000` (in TrainConfig) means we checkpoint every 5000 steps.
At the rates above that's every ~3-6 hours.

## What we deliberately *did not* set up (and why)

These appear in the design doc's "Out of Scope" list. Repeating here for
the operations perspective:

- **DDP** (multi-GPU on one box): not needed for 124M on one 5090. The
  loader and training loop are written rank-aware so adding it later is
  ~5 lines plus a `torchrun` launch.
- **FSDP / ZeRO / pipeline / tensor parallel**: for models that don't fit
  on one GPU. 124M fits 100x over. Not relevant here.
- **DeepSpeed / Accelerate / Lightning / Fabric**: framework wrappers. They
  hide what's happening in the training step behind APIs. We want to see
  every step — pedagogically vital, operationally simpler.
- **`torch.compile` cache management**: TorchInductor caches compiled graphs
  to `~/.cache/torch/inductor/`. Across PyTorch upgrades this can grow or
  go stale. For one project on one machine we ignore it; if it ever gets
  in the way, `rm -rf ~/.cache/torch/inductor/`.
- **Checkpoint compression / safetensors**: plain `.pt` is fine. ~500 MB per
  checkpoint, ~4 checkpoints across a run.
- **A `Trainer` class abstraction**: hides the loop. Don't want that here.
- **LR finder**: we use the GPT-2 paper's published LR, which is known to
  work.
- **EMA weights**: not used in standard GPT-2 training.

## When to graduate to a "real" setup

If you start training a 7B model, or do multi-machine training, or want
hyperparameter sweeps, you'll need to revisit the choices above. By that
point, the right move is usually to fork into a separate project (or use
Karpathy's `nanotron` / `litgpt` / etc.) rather than retrofit this one.

The goal here was "understand the mechanics," not "production training
platform." Once you understand the mechanics, the frameworks make sense.
```

- [ ] **Step 2: Commit**

```bash
git add docs/04-hardware-5090.md
git commit -m "docs: 5090 hardware notes, tuning playbook, what we skip"
```

---

## Task 15: `docs/05-eval-and-sampling.md`

**Files:**
- Create: `docs/05-eval-and-sampling.md`

- [ ] **Step 1: Write the doc**

Create `docs/05-eval-and-sampling.md`:

```markdown
# Eval and Sampling

Three ways to look at how the model is doing:

1. **Val loss** — quantitative, immediate, runs every `val_interval` steps.
2. **HellaSwag** — quantitative, slower, runs every `hella_interval` steps.
3. **Generated samples** — qualitative, run manually with `sample.py`.

## Val loss

Held out: shard 0 of FineWeb-Edu (`edufineweb_val_000000.bin`, ~100M tokens).
Never seen during training.

`train.py::_evaluate` runs `cfg.val_iters` validation batches under
`torch.no_grad()` and `model.eval()`, averages the cross-entropy. Logged
to `train.csv` as `kind=val`.

What to expect for GPT-2 124M on FineWeb-Edu:
- Start of training: ~10 (random init, log(vocab) ≈ log(50304) ≈ 10.8).
- After warmup (~step 1000): ~5-6.
- End of training: ~3.0-3.3, depending on data quality.

Lower is better. The gap between train and val loss tells you about
overfitting — for pre-training at this scale, the gap should be tiny (we're
training on 10B tokens once each).

## HellaSwag (zero-shot)

`eval_hellaswag.py::evaluate_hellaswag`.

What it measures: commonsense plausibility of sentence completions.

The model never sees HellaSwag during training. At eval time, we present
the 4 candidate endings as completions and pick the lowest-NLL one.

Scoring:
```
score(ending) = mean(per-token NLL of ending tokens given context)
prediction = argmin over 4 endings
```

Baselines for orientation:
- Random: 25%
- GPT-2 124M (paper): 28.9%
- GPT-2 medium (350M): 33.7%
- GPT-3 175B (zero-shot): 78.9%
- Human: 95.6%

We expect our 124M to land around 28-30%. If we're below 26% deep into
training, something is wrong (likely a bug in scoring, not the model).

## Generated samples

Run `sample.py`:

```bash
uv run python sample.py --ckpt checkpoints/model_010000.pt \
    --prompt "Hello, I'm a language model," \
    --max_tokens 200 --n_samples 3 --temperature 0.8 --top_k 50
```

Knobs:

- `--temperature`: > 1 = more random, < 1 = more deterministic. 0.8 is a
  decent default.
- `--top_k`: at each step, only sample from the K highest-prob tokens.
  Reduces "long tail" weirdness. 50 is common.

At GPT-2 124M scale you'll see grammatically correct but topically incoherent
text. That's expected — 124M is not enough capacity for strong coherence.

## Putting it together

A healthy training run looks like:

- Train loss: smoothly decreasing in log space, roughly straight on a
  log-y plot, with a flat plateau forming toward the end.
- Val loss: tracks train loss closely (within ~0.1 at this scale).
- HellaSwag: noisy but trending up from 25% toward 29-30% over the run.
- Generated samples: nonsense at step 1000, locally coherent by step 5000,
  topically loose-but-readable by the end.

A pathological run looks like:

- Train loss spikes / NaN: grad clipping should catch this, but if it
  doesn't, the LR is too high. Lower max_lr or reduce warmup tokens.
- Train loss flat at log(vocab) ~= 10.8 forever: the model isn't training.
  Either grads aren't flowing (bug) or the LR is effectively zero.
- Val loss diverges from train loss: shouldn't happen at this scale on 10B
  tokens, but if it does, your data is funky.
- HellaSwag stuck at 25%: scoring bug.
```

- [ ] **Step 2: Commit**

```bash
git add docs/05-eval-and-sampling.md
git commit -m "docs: eval and sampling walkthrough"
```

---

## Task 16: Final `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the placeholder README**

Overwrite `README.md`:

```markdown
# minimal-gpt2

A minimal, heavily commented training project for GPT-2 124M, for learning
the mechanics of modern transformer pre-training. Follows Karpathy's
`build-nanogpt` (the 2024 "Reproducing GPT-2" video).

## What's in here

- **Model**: vanilla GPT-2 124M (12 layers, 12 heads, 768 dim, 1024 context).
- **Data**: FineWeb-Edu `sample-10BT` (~10B tokens), tokenized to `uint16` shards.
- **Training**: single-GPU loop, bf16 + `torch.compile`, cosine LR, AdamW, grad accum.
- **Eval**: validation loss + zero-shot HellaSwag.
- **Workflow**: dev/test on Mac, train on a Linux box with one RTX 5090.

Code is structured for *reading*. Every important choice has an inline
comment explaining why. The `docs/` directory has block-by-block walkthroughs
of the model, data pipeline, training recipe, hardware setup, and eval.

## Repo layout

```
.
├── docs/
│   ├── 01-architecture.md
│   ├── 02-data-pipeline.md
│   ├── 03-training-recipe.md
│   ├── 04-hardware-5090.md
│   └── 05-eval-and-sampling.md
├── model.py            # GPT-2 model
├── config.py           # GPTConfig + TrainConfig
├── data.py             # Sharded DataLoaderLite
├── train.py            # Training loop
├── sample.py           # Generate from a checkpoint
├── eval_hellaswag.py   # Zero-shot HellaSwag
├── utils.py            # LR schedule, device, CSV logger, seeding
├── scripts/
│   ├── prep_shakespeare.py    # Mac smoke-test dataset
│   └── prep_fineweb_edu.py    # Full 10B tokenization (run on 5090)
├── tests/              # Unit tests + smoke train
├── data/               # gitignored — token shards
├── logs/               # gitignored — train.csv
└── checkpoints/        # gitignored — model_*.pt
```

## Quickstart

### On Mac (development)

```bash
# Install (uv reads .python-version and pyproject.toml).
uv sync --extra dev

# Generate tiny smoke-test data.
uv run python scripts/prep_shakespeare.py

# Run all tests including the smoke train (~30s on M-series Mac).
uv run pytest -v
```

### On the 5090 workstation (real training)

```bash
# Same install on Linux (uv handles CUDA wheel selection for torch).
uv sync --extra dev

# 1) Tokenize FineWeb-Edu 10B (~few hours).
uv run python scripts/prep_fineweb_edu.py

# 2) Quick canary run — 50 steps. Watch nvidia-smi + tokens/sec.
uv run python train.py --max_steps 50 --val_interval 100 --hella_interval 100 --save_interval 100

# 3) Tune micro_batch_size + grad_accum_steps (see docs/04-hardware-5090.md).

# 4) Full run.
uv run python train.py

# Resume if needed.
uv run python train.py --resume checkpoints/model_005000.pt

# Sample from a checkpoint.
uv run python sample.py --ckpt checkpoints/model_015000.pt --prompt "Hello, I'm a language model,"
```

## Reading order

If you're new to the codebase:

1. `docs/01-architecture.md` + `model.py` — the model.
2. `docs/02-data-pipeline.md` + `data.py` + `scripts/prep_fineweb_edu.py` — the data.
3. `docs/03-training-recipe.md` + `train.py` — the training loop.
4. `docs/04-hardware-5090.md` — running it for real.
5. `docs/05-eval-and-sampling.md` + `eval_hellaswag.py` + `sample.py` — looking at results.

## Acknowledgements

This project is a learning vehicle, not original work. It's a re-creation of
Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) with
extra inline commentary aimed at someone learning transformer pre-training.

## License

For personal learning use. Not intended for distribution.
```

- [ ] **Step 2: Verify the full test suite still passes**

Run:
```bash
uv run pytest -v
```
Expected: all tests pass (test_config, test_model, test_utils, test_data, test_smoke_train).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: final README with quickstart, layout, and reading order"
```

---

## Final Verification

- [ ] **Step 1: Run the full test suite one more time**

```bash
uv run pytest -v
```
Expected: every test passes.

- [ ] **Step 2: Run ruff to catch any style issues**

```bash
uv run ruff check .
```
Expected: no issues, or trivial ones to fix in-line and re-commit.

- [ ] **Step 3: Confirm git state is clean**

```bash
git status
git log --oneline
```
Expected: clean working tree, one commit per task.

The project is now ready to be moved to the 5090 workstation for data prep
and full training. The expected next steps live in `docs/04-hardware-5090.md`.
