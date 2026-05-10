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
