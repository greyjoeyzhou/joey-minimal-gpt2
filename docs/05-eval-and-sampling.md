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
