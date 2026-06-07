# minimal-gpt2

A minimal, heavily commented training project for learning the mechanics of
modern transformer pre-training. Started as a re-creation of Karpathy's
`build-nanogpt`, then evolved through four model versions — from vanilla GPT-2
up to a miniature DeepSeek V2/V3-style architecture.

All versions train on **FineWeb-Edu `sample-10BT`** (~10B tokens) on a single
RTX 5090. Dev/test runs on Mac.

---

## Model versions

| | v0 (GPT-2) | v1 (modern) | v2 (nanowhale) | v3 (v1 + MoE) |
|---|---|---|---|---|
| **File** | `model.py` | `model_v1.py` | `model_v2.py` | `model_v3.py` |
| **Params (total)** | ~124M | ~114M | ~106M | ~155M |
| **Active/token** | ~124M | ~114M | ~60M | ~116M |
| **Layers** | 12 | 12 | 8 | 12 |
| **Norm** | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| **Position** | Learned abs. | RoPE | RoPE + NoPE | RoPE |
| **Attention** | MHA | GQA | MLA | GQA |
| **FFN** | GELU dense | SwiGLU dense | SwiGLU MoE | SwiGLU MoE |
| **Residual** | Standard | Standard | Hyper-Connections | Standard |
| **Extra** | — | — | MTP aux head | load-balance loss |
| **Context** | 1024 | 2048 | 2048 | 2048 |
| **HellaSwag** | 30.6% | 30.2% | not trained | TBD |
| **Trained to** | step 29,999 | step 19,072 | — | step 19,072 |

**v2** is a miniature DeepSeek V2/V3-inspired model: MLA attention + DeepSeekMoE
(shared + routed experts) + Multi-Token Prediction, with Hyper-Connections
(from nanowhale) replacing standard residuals. It has not been trained yet.

See `ARCHITECTURE_NOTES.md` for a detailed component-by-component breakdown
and `MOE_DISPATCH_ITERATIONS.md` for the v3 MoE dispatch engineering story.

---

## Speed benchmarks (RTX 5090, torch.compile, micro_batch=16, seq_len=2048)

| | v1 (modern) | v2 (nanowhale) | v3 (v1 + MoE) |
|---|---|---|---|
| **ms / step** | 172.4 ± 0.1 | 222.9 ± 0.5 | 238.2 ± 0.1 |
| **tok / s** | 190,107 | 146,989 | 137,593 |
| **vs v1** | baseline | 1.29× slower | 1.38× slower |

v0 not benchmarked (different seq_len=1024; roughly comparable to v1).

The MoE overhead (v2, v3) is real but bounded: both models use a fixed-shape
dispatch that compiles once. The near-zero step variance confirms no per-step
recompilation. See `MOE_DISPATCH_ITERATIONS.md` for the engineering story.
Run `bench.py` to reproduce.

---

## Repo layout

```
.
├── model.py / model_v1.py / model_v2.py / model_v3.py   # model definitions
├── config.py / config_v1.py / config_v2.py / config_v3.py
├── train.py / train_v1.py / train_v2.py / train_v3.py
├── eval_hellaswag.py / eval_hellaswag_v1.py / ...
├── sample.py / sample_v1.py / sample_v2.py / sample_v3.py
├── data.py                      # shared sharded DataLoaderLite
├── utils.py                     # LR schedule, device, CSV logger, seeding
├── ARCHITECTURE_NOTES.md        # all-versions architecture reference
├── MOE_DISPATCH_ITERATIONS.md   # v3 MoE dispatch: 3 implementations explained
├── docs/
│   ├── 01-architecture.md
│   ├── 02-data-pipeline.md
│   ├── 03-training-recipe.md
│   ├── 04-hardware-5090.md
│   └── 05-eval-and-sampling.md
├── scripts/
│   ├── prep_shakespeare.py      # Mac smoke-test dataset
│   └── prep_fineweb_edu.py      # Full 10B tokenization (run on 5090)
├── tests/
├── data/          # gitignored — token shards
├── logs/          # gitignored — v0 train.csv
├── logs_v1/       # gitignored — v1 train.csv
├── logs_v3/       # gitignored — v3 train.csv
├── checkpoints/   # gitignored — v0 model_*.pt
├── checkpoints_v1/
└── checkpoints_v3/
```

---

## Quickstart

### On Mac (development)

```bash
uv sync --extra dev
uv run python scripts/prep_shakespeare.py   # tiny smoke-test data
uv run pytest -v                            # unit tests + smoke train (~30s)
```

### On the 5090 (real training)

```bash
uv sync --extra dev

# Tokenize FineWeb-Edu 10B (one-time, takes a few hours).
uv run python scripts/prep_fineweb_edu.py

# Train — replace train.py with train_v1.py / train_v3.py for other versions.
uv run python train.py

# Resume from a checkpoint.
uv run python train.py --resume checkpoints/model_010000.pt

# Evaluate on HellaSwag.
uv run python eval_hellaswag.py --ckpt checkpoints/model_019072.pt

# Sample from a checkpoint.
uv run python sample.py --ckpt checkpoints/model_019072.pt --prompt "Hello, I'm a language model,"
```

---

## Reading order

1. `docs/01-architecture.md` + `model.py` — start here (vanilla GPT-2).
2. `model_v1.py` — adds RMSNorm, RoPE, GQA, SwiGLU.
3. `model_v3.py` + `MOE_DISPATCH_ITERATIONS.md` — adds MoE on top of v1; dispatch engineering story.
4. `model_v2.py` — full DeepSeek-style stack (MLA + MoE + Hyper-Connections + MTP).
5. `ARCHITECTURE_NOTES.md` — all components explained side by side.
6. `docs/02-data-pipeline.md`, `docs/03-training-recipe.md`, `docs/04-hardware-5090.md` — data, training, hardware.

---

## Acknowledgements

Built on top of Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt).
v2 architecture inspired by [nanowhale](https://huggingface.co/HuggingFaceTB/nanowhale-100m-base)
and the DeepSeek V2/V3 papers.

## License

Personal learning use. Not intended for distribution.
