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
