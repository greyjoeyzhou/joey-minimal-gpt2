# Abbreviations

Cheat-sheet for every abbreviation in this codebase (and most you'll see in the broader transformer/LLM literature). Grouped by category. When in doubt about a name in `model.py` / `config.py` / `train.py`, look here first.

See also: [`embedding.md`](./embedding.md), [`tokenizer.md`](./tokenizer.md), [`training-data.md`](./training-data.md) for deeper coverage of specific topics.

## Model architecture components

| Abbreviation | Stands for | What it is |
|---|---|---|
| **LN** | **L**ayer **N**orm | Normalizes activations within a layer (mean=0, var=1, then scale+shift). Pre-LN (LN *before* the sublayer) is what makes deep transformers trainable. |
| **ln_1** | First **L**ayer**N**orm in a block | Applied before attention |
| **ln_2** | Second **L**ayer**N**orm in a block | Applied before MLP |
| **ln_f** | **F**inal **L**ayer**N**orm | Applied once after the last block, before `lm_head` |
| **LM** | **L**anguage **M**odel | `lm_head` = the output projection that turns hidden states into vocab logits |
| **MLP** | **M**ulti-**L**ayer **P**erceptron | The position-wise feedforward block. Two linear layers + GELU |
| **FFN** | **F**eed-**F**orward **N**etwork | Same thing as MLP. Different papers use different names. |
| **attn** | **att**e**n**tion | Self-attention module |
| **h** | **h**idden / **h**eads | `transformer.h` is the `ModuleList` of transformer blocks. Yes, just `h`. (HuggingFace GPT-2 naming.) |

## Inside attention

| Abbreviation | Stands for | What it is |
|---|---|---|
| **Q, K, V** | **Q**uery, **K**ey, **V**alue | The three projections inside attention. Q asks, K is indexed by, V is what gets retrieved |
| **qkv** | concatenation of Q, K, V | The output of `c_attn`, before being split |
| **c_attn** | "**c**ombined attention" (originally `c` = `Conv1D`) | Fused QKV projection — one matrix that produces all three of Q, K, V at once |
| **c_proj** | "**c**ombined projection" | The output projection of a sublayer (attention or MLP) |
| **c_fc** | "**c**ombined **f**ully-**c**onnected" | The MLP's first linear (expansion to 4×) |
| **n_head** | **n**umber of attention **head**s | = 12 for us. Each head gets `n_embd / n_head` = 64 dims |
| **head_dim** | dimension **per head** | = `n_embd / n_head` = 64 |

**Naming quirk worth knowing:** the `c_` prefix in `c_attn`, `c_proj`, `c_fc` is historical baggage from OpenAI's TF1 code, where these linear layers were implemented using a 1D convolution (`Conv1D`) for reasons that no longer matter. The "c" stands for "Conv1D." Karpathy kept the names in nanoGPT for compatibility with HuggingFace's checkpoint loading. **Read `c_` as "this is just a linear layer."**

## Embeddings & tokenization

| Abbreviation | Stands for | What it is |
|---|---|---|
| **wte** | **W**ord **T**oken **E**mbedding | Token ID → vector lookup table. (50304, 768) for us |
| **wpe** | **W**ord **P**ositional **E**mbedding | Position → vector lookup table. (1024, 768) for us |
| **BPE** | **B**yte-**P**air **E**ncoding | The tokenization algorithm — see [`tokenizer.md`](./tokenizer.md) |
| **EOT** | **E**nd **O**f **T**ext | The `<|endoftext|>` special token, ID 50256 |
| **vocab_size** | size of the vocabulary | = 50304 for us (50257 real + padding to multiple of 64) |
| **block_size** | maximum context length | = 1024 for us. The "T" dimension is capped by this |

**Naming quirk:** `wte` and `wpe` use "word" even though GPT-2 operates on BPE sub-word tokens, not words. Historical baggage — the names predate BPE. Read as "token" and "position."

## Tensor dimensions

The standard shape tuple in this codebase (and most PyTorch transformer code) is `(B, T, C)`. See `embedding.md` §3 for full detail.

| Abbreviation | Stands for | Default | Where defined |
|---|---|---|---|
| **B** | **B**atch | 32 | `micro_batch_size`, `config.py:74` |
| **T** | **T**ime (= sequence length) | 1024 | `seq_len`, `config.py:77` |
| **C** | **C**hannels (= embedding dim) | 768 | `n_embd`, `config.py:49` |
| **V** | **V**ocab size | 50304 | `vocab_size`, `config.py:39` |
| **n_embd** | **n**umber of **embed**ding dimensions | 768 | Same as `C`, same as `d_model` |
| **n_layer** | **n**umber of transformer **layer**s | 12 | Depth |
| **d_model** | model dimension | (= n_embd) | Term from the original Attention Is All You Need paper. Same number. |
| **hidden_size** | another name for n_embd | (= n_embd) | HuggingFace's convention |

**Why so many names for the same thing (`C` = `n_embd` = `d_model` = `hidden_size`):** different communities settled on different names. Karpathy uses `C`. The original transformer paper uses `d_model`. HuggingFace configs use `n_embd` or `hidden_size`. They all mean the same number: the dimension every token vector has throughout the network. = 768 for GPT-2 small.

## Activations & normalization

| Abbreviation | Stands for | What it is |
|---|---|---|
| **GELU** | **G**aussian **E**rror **L**inear **U**nit | The activation function inside MLP. Smoother variant of ReLU. We use the tanh approximation (`approximate="tanh"`) to match GPT-2 |
| **ReLU** | **Re**ctified **L**inear **U**nit | `max(0, x)`. Older alternative to GELU |
| **SwiGLU** | **Swi**sh-**G**ated **L**inear **U**nit | Modern activation used by LLaMA, PaLM. Adds a gating mechanism to the MLP |
| **LN** | **L**ayer **N**orm | See above |
| **RMSNorm** | **R**oot **M**ean **S**quare **Norm**alization | Simpler/faster variant of LayerNorm used by LLaMA-1+. Skips the mean-subtraction step |

## Training-related

| Abbreviation | Stands for | What it is |
|---|---|---|
| **LR / lr** | **L**earning **R**ate | Step size for gradient descent. `max_lr=6e-4`, `min_lr=6e-5` for us |
| **AdamW** | **Adam** with decoupled **W**eight decay | Our optimizer. Fixes a subtle bug in Adam's L2 regularization (Loshchilov & Hutter, 2017) |
| **SGD** | **S**tochastic **G**radient **D**escent | Old-school baseline optimizer; not used here |
| **DDP** | **D**istributed **D**ata **P**arallel | PyTorch's multi-GPU training mode where each GPU has a copy of the model and processes different data |
| **FSDP** | **F**ully **S**harded **D**ata **P**arallel | Modern alternative to DDP for very large models — shards optimizer state, gradients, and parameters across GPUs |
| **ckpt** | **c**heckpoint | Saved model state |
| **CE loss** | **C**ross-**E**ntropy loss | The standard classification loss; what we use for next-token prediction |
| **bf16** | **b**rain**f**loat **16** | 16-bit float with same exponent range as fp32. Modern default for training |
| **fp16** | **f**loating **p**oint **16** | 16-bit float with reduced range. Older, requires loss scaling |
| **fp32** | **f**loating **p**oint **32** | Standard 32-bit float. Master copy lives in fp32; bf16 used for forward/backward |
| **grad_clip** | **grad**ient **clip**ping | Cap on gradient norm to prevent training instability |
| **grad_accum** | **grad**ient **accum**ulation | Run multiple forward+backward passes before each optimizer step, to simulate a larger batch on limited VRAM |

## Modern attention variants (post-GPT-2)

You won't see these in this codebase, but you'll see them constantly in the LLaMA/Mistral/etc. literature.

| Abbreviation | Stands for | What it is |
|---|---|---|
| **MHA** | **M**ulti-**H**ead **A**ttention | Standard attention (what we use). N heads, each with its own Q, K, V |
| **MQA** | **M**ulti-**Q**uery **A**ttention | N query heads but **1 shared K/V head**. Reduces KV cache memory ~Nx. Used by PaLM, Falcon |
| **GQA** | **G**rouped-**Q**uery **A**ttention | Middle ground: N query heads, G shared K/V heads (G < N). Used by LLaMA-2/3, Mistral. The current default for most open models |
| **MLA** | **M**ulti-head **L**atent **A**ttention | DeepSeek-V2's innovation. Compresses K/V into a small latent space. Even better than GQA for inference memory |
| **SWA** | **S**liding **W**indow **A**ttention | Each token only attends to the last W tokens (not all previous). Used by Mistral. Trades global context for `O(T)` instead of `O(T²)` |

## Position encoding variants

| Abbreviation | Stands for | What it is |
|---|---|---|
| **APE** | **A**bsolute **P**osition **E**ncoding | What we use (`wpe`). Learned vector per position |
| **RPE** | **R**elative **P**osition **E**ncoding | Encodes the *distance* between tokens, not absolute position |
| **RoPE** | **Ro**tary **P**osition **E**mbedding | Modern standard. Rotates Q and K by position-dependent angles. Used by LLaMA, Mistral, Qwen, etc. |
| **ALiBi** | **A**ttention with **Li**near **Bi**ases | Adds linear distance bias to attention scores. Used by BLOOM, MPT |
| **NoPE** | **No** **P**osition **E**ncoding | Recent research showing causal masking alone provides enough positional signal |

## Algorithms / techniques

| Abbreviation | Stands for | What it is |
|---|---|---|
| **MLM** | **M**asked **L**anguage **M**odeling | BERT-style: randomly mask tokens, predict them. Bidirectional |
| **CLM** | **C**ausal **L**anguage **M**odeling | GPT-style: predict next token given previous. Autoregressive. **What we do.** |
| **SFT** | **S**upervised **F**ine-**T**uning | Fine-tuning a base model on (input, target output) pairs |
| **RLHF** | **R**einforcement **L**earning from **H**uman **F**eedback | Aligning models with human preferences via RL |
| **DPO** | **D**irect **P**reference **O**ptimization | Simpler alternative to RLHF — no separate reward model |
| **LoRA** | **Lo**w-**R**ank **A**daptation | Parameter-efficient fine-tuning: train small rank-decomposed updates instead of full weights |
| **PEFT** | **P**arameter-**E**fficient **F**ine-**T**uning | Umbrella term for LoRA-style techniques |
| **MoE** | **M**ixture **o**f **E**xperts | Architecture where different MLPs ("experts") are activated for different tokens. Used by Mixtral, DeepSeek-V3, GPT-4 (rumored) |
| **FIM** | **F**ill-**I**n-the-**M**iddle | Training objective for code models — predict the middle given prefix + suffix |
| **KV cache** | **K**ey-**V**alue cache | Inference-time optimization: cache K and V for previous tokens so generation is `O(T)` per token, not `O(T²)` |

## Evaluation

| Abbreviation | Stands for | What it is |
|---|---|---|
| **PPL** | **P**er**pl**exity | `exp(cross_entropy_loss)`. Lower = better. Standard LM eval metric |
| **HellaSwag** | (not an abbreviation, a benchmark name) | Multiple-choice commonsense reasoning. What `eval_hellaswag.py` runs |
| **MMLU** | **M**assive **M**ultitask **L**anguage **U**nderstanding | 57-subject academic knowledge benchmark |
| **ARC** | **A**I2 **R**easoning **C**hallenge | Grade-school science questions |
| **GSM8K** | **G**rade-**S**chool **M**ath **8K** | 8k arithmetic word problems |
| **HumanEval** | (benchmark name) | Code-completion benchmark — 164 hand-written Python problems |

## Hardware / data-formats

| Abbreviation | Stands for | What it is |
|---|---|---|
| **GPU** | **G**raphics **P**rocessing **U**nit | NVIDIA hardware we train on |
| **VRAM** | **V**ideo **RAM** | GPU memory. 32 GB on a 5090 |
| **HBM** | **H**igh **B**andwidth **M**emory | The memory tech inside GPUs |
| **SRAM** | **S**tatic **RAM** | On-chip cache, much smaller but much faster than HBM. Flash Attention exploits this |
| **TFLOPS** | **T**era **FL**oating **P**oint **O**perations **P**er **S**econd | Compute throughput |
| **MFU** | **M**odel **FL**OPs **U**tilization | What fraction of theoretical peak compute we actually achieve. 30-50% is good |
| **WARC** | **W**eb **AR**chive (file format) | The format CommonCrawl distributes their crawls in |
| **WAL** | **W**rite-**A**head **L**og | Not LLM-specific but appears in checkpointing context |

## Convention summary

A few patterns worth internalizing:

- **`n_*`** prefix (`n_embd`, `n_head`, `n_layer`) = "number of X." NanoGPT/HuggingFace convention.
- **`c_*`** prefix (`c_attn`, `c_proj`, `c_fc`) = legacy `Conv1D` naming. Just means "linear layer."
- **`ln_*`** = LayerNorm. `ln_1`, `ln_2` are positions within a block; `ln_f` is final.
- **`w*e`** = embedding tables (`wte`, `wpe`).
- **Lowercase single letters** (`h`, `x`, `y`, `B`, `T`, `C`) = either module names (`h`) or tensor dimension labels.
- **`max_*`, `min_*`** = bounds (`max_lr`, `min_lr`, `max_steps`).
- **`*_interval`, `*_steps`** = scheduling (`val_interval`, `warmup_steps`).
