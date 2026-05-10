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
