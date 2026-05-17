# Embeddings

Notes on the embedding layer of our GPT-2 implementation — the full input pipeline (text → tokens → embeddings), the tensor-shape conventions, what `wte` and `wpe` are, how they're learned, where they live in the code, and what modern alternatives look like.

## 1. The pipeline: text → tokens → embeddings

A transformer doesn't take text as input. It takes a tensor of integers. The embedding layer is the *bridge* between integer token IDs and the continuous vector space the rest of the network operates in. The full pipeline:

```
"Hello world, how are you?"          ← Python string (raw text)
        │
        │  tiktoken.encoding_for_model("gpt2").encode(text)
        ▼
[15496, 995, 11, 703, 389, 345, 30]  ← list of int, length T (here T=7)
        │
        │  torch.tensor(...).unsqueeze(0)  →  shape (B=1, T=7), dtype int64
        ▼
idx: (B, T)                          ← this is what model.forward(idx) takes
        │
        │  wte(idx)        →  tok_emb: (B, T, n_embd)
        │  wpe(arange(T))  →  pos_emb:    (T, n_embd)
        │  add them
        ▼
x: (B, T, n_embd)                    ← input to the first transformer block
```

This pipeline runs at two very different times in the two modes:

- **Training (offline tokenization)**: text is tokenized *once*, ahead of time, by a prep script in `scripts/` using `tiktoken` (OpenAI's fast BPE library). The result is written to disk as `.bin` files of `uint16` token IDs. During training, `DataLoaderLite` just reads those bytes off disk and reshapes — *no tokenizer runs in the training loop*. See `data.py:37`: `np.fromfile(path, dtype=np.uint16)`. The tokenization cost is amortized to zero across every epoch.
- **Inference / sampling (online tokenization)**: when generating text from a prompt, the tokenizer runs on the prompt string at inference time, you get back the token IDs, and *those* go through `wte` + `wpe`. See `sample.py`.

The tokenizer and the model are **tightly coupled but separately built**. The tokenizer is *not* trained jointly with the model — it's pre-built (in our case, downloaded from OpenAI via `tiktoken`) and frozen for the entire life of the model. The model only learns what each token ID *means* (via `wte`); it never has any say in how text gets chunked into IDs in the first place. Swap tokenizers → the trained model speaks gibberish.

## 2. BPE tokenization — what `idx` actually contains

We use **BPE (Byte-Pair Encoding)** — specifically, OpenAI's GPT-2 BPE tokenizer. Each integer in `idx` represents a *sub-word* unit, not a character. This is why `vocab_size = 50,257` (the real GPT-2 vocab; we pad to 50,304 for tensor-core alignment). If it were character-level, vocab size would be ~256 (one per byte) or ~100k (one per Unicode codepoint), and sequences would be much, much longer.

To get a feel for how BPE chunks text, here's what GPT-2's tokenizer does:

| Text | Tokens | Token IDs |
|---|---|---|
| `"hello world"` | `["hello", " world"]` | `[31373, 995]` |
| `"unbelievable"` | `["un", "bel", "iev", "able"]` | `[403, 6667, 11203, 540]` |
| `"GPT-2"` | `["G", "PT", "-", "2"]` | `[38, 11571, 12, 17]` |
| `"日本"` | (split into UTF-8 bytes, each byte a token) | `[33768, 98, 17312, 105]` |

So a single token can be:
- a whole common word (`"hello"`)
- a word with leading space (`" world"` — note the leading space *is part of the token*; GPT-2 BPE encodes spaces this way, which is why generation handles word boundaries cleanly)
- a fragment of a rare word (`"un"`, `"bel"`)
- a single character for rare symbols
- a single UTF-8 byte for languages it wasn't trained on much

**Why BPE and not chars?** Two reasons:

1. **Sequence length.** English averages ~4 characters per BPE token, so BPE makes sequences ~4× shorter than char-level. Attention is `O(T²)` in time and memory, so 4× shorter sequences are 16× cheaper. At our `block_size=1024`, you fit about 750–800 words of English — a meaningful chunk of context. Char-level at the same compute budget would fit maybe 200 words.
2. **Useful units.** "hello" carrying meaning as one indivisible vector is more learnable than the model having to reconstruct the meaning of "hello" from 5 character vectors and learn that those 5 chars together have meaning distinct from the chars in "yellow."

**The tradeoff** is that BPE has known weaknesses: it can't easily count letters in a word ("how many r's in strawberry?" — it sees `["str", "aw", "berry"]`, no individual `r`s), it tokenizes numbers in inconsistent ways (`"1234"` might be `["12", "34"]` or `["1", "234"]` depending on context), and it handles non-English text poorly (the Japanese example above — falling back to byte-level is expensive in tokens). Modern models address some of this by training on much larger, more multilingual tokenizers (LLaMA-3 uses 128k vocab) or by exploring byte-level / patch-level alternatives (Meta's BLT paper, 2024).

## 3. Tensor shape conventions: `idx`, `B`, `T`, `C`

`idx` is short for **indices** — specifically, the integer token IDs that index into the vocabulary. It's a naming convention from `nn.Embedding`, whose input is "arbitrary shape containing the indices to extract." Karpathy uses `idx` throughout nanoGPT and it stuck.

So when you see `wte(idx)`, what's literally happening is: `idx[b, t]` is an integer like `15496`, and `wte(idx)[b, t]` is the row 15496 of the `wte` weight matrix — the 768-dim vector representing whatever token has ID 15496. **Embedding lookup is just fancy indexing into a 2D tensor**, nothing more.

The standard shape tuple you'll see all over this codebase (and nanoGPT, and most PyTorch transformer code) is `(B, T, C)`:

| Letter | Meaning | What it is here | Default value |
|---|---|---|---|
| **B** | **Batch** | Number of independent sequences processed in parallel in one forward pass | `micro_batch_size = 32` (`config.py:74`) |
| **T** | **Time** (= sequence length, = number of tokens) | How many tokens long each sequence is | `seq_len = 1024` (`config.py:77`) |
| **C** | **Channels** (= embedding dim, = `n_embd`, = `d_model`) | Vector size used to represent each token throughout the network | `n_embd = 768` (`config.py:49`) |

A few naming notes:

- `T` stands for **Time**, a holdover from RNN/sequence-modeling days when sequences were processed one "time step" at a time. The convention stuck even after transformers replaced RNNs.
- `C` for "channels" is borrowed from CNNs (where an image tensor is `(B, C, H, W)`). You'll see all three names — `C`, `n_embd`, `d_model` — referring to the same number (768 here). Karpathy uses `C`; the Attention Is All You Need paper uses `d_model`; HuggingFace configs use `n_embd` or `hidden_size`.
- `V` for vocab size shows up at the output, where logits live: `(B, T, V)`.

Full picture of all shapes for one training step (defaults: B=32, T=1024, C=768, V=50304):

```
idx:     (B=32, T=1024)              token IDs, int64
tok_emb: (B=32, T=1024, C=768)       float32/bf16, output of wte(idx)
pos_emb: (T=1024, C=768)             float32/bf16, output of wpe(arange(T)) — no B dim!
x:       (B=32, T=1024, C=768)       sum of the two, broadcast over batch
                                     flows through every block unchanged in shape
logits:  (B=32, T=1024, V=50304)     float32, output of lm_head
```

`pos_emb` having no `B` dim is the "same positions for every sequence in the batch" thing — PyTorch broadcasting aligns from the right, so `(T, C) + (B, T, C)` adds the same positional vector to every sequence's tokens.

How `(B, T)` is constructed by the data loader (`data.py:119-120`):
```python
x = buf[:-1].view(B, T)   # (B, T) = (32, 1024) = 32,768 token IDs
y = buf[1:].view(B, T)    # same buffer shifted by one → next-token targets
```

The loader reads `B*T + 1` consecutive `int64` token IDs from disk as a flat 1D tensor, then reshapes the first `B*T` into `(B, T)` for `x` (= the `idx` fed to the model) and the same buffer shifted by one into `(B, T)` for `y` (= the targets). Nothing in this reshape respects sentence or document boundaries — it's just a contiguous slice of the corpus. Document boundaries are encoded as the `<|endoftext|>` token during preprocessing; the model sees this as one long sequence and learns from it.

## 4. What do `wte` and `wpe` stand for?

These names come straight from the original HuggingFace GPT-2 implementation (they kept the names from OpenAI's TensorFlow checkpoint):

- **`wte`** = **W**ord **T**oken **E**mbedding — the lookup table that maps each token ID (an integer in `[0, vocab_size)`) to a vector of size `n_embd`.
- **`wpe`** = **W**ord **P**ositional **E**mbedding — the lookup table that maps each position (an integer in `[0, block_size)`) to a vector of size `n_embd`.

Why "word" when GPT-2 actually operates on BPE sub-word tokens? Historical baggage — the names predate the move to BPE and just stuck. Read them as "token" and "position." This naming matters because if you ever want to load HuggingFace's pretrained GPT-2 weights into your model, the parameter names have to match exactly. That's why `model.py` uses `nn.ModuleDict(dict(wte=..., wpe=..., h=..., ln_f=...))` rather than nicer Pythonic names.

## 5. Are they learned or pre-existing?

**Both are learned from scratch** during training. They're plain `nn.Embedding` modules — which under the hood is just a 2D parameter tensor (`weight` of shape `(num_embeddings, embedding_dim)`) plus an index-lookup op. At init time they're filled with `N(0, 0.02)` random values, and every optimizer step updates them via the gradient of the loss with respect to the rows that got used in that batch.

Two important details about how they get updated:

- **Token embeddings get gradients sparsely**: only the rows for token IDs that actually appeared in the batch receive a gradient that step. If token 42 never shows up in your training data, its row never moves from its random init. (At 10B-token scale this isn't a worry — every common token gets hit constantly.)
- **The token embedding table is *weight-tied* to the LM head** (`model.py:177`), which means it gets a *second* gradient signal per step: one from being used as input embeddings, and one from being used as the output projection that produces logits. Weight tying is covered as a separate topic, but it's worth knowing now because it changes the gradient dynamics for `wte` specifically.

You *can* initialize from HuggingFace's pretrained checkpoint instead (nanoGPT has a `from_pretrained` classmethod for this) — but that's a fine-tuning workflow, not what `train.py` here does. This repo trains from scratch on FineWeb-Edu.

## 6. Where in the code?

Three places matter:

**Definition** — `model.py:160-161`:
```python
wte=nn.Embedding(config.vocab_size, config.n_embd),   # (50304, 768)
wpe=nn.Embedding(config.block_size, config.n_embd),   # (1024, 768)
```

With the default `GPTConfig` (`config.py:32,39,49`), `wte` is a `(50304, 768)` matrix = ~38.6M params, and `wpe` is a `(1024, 768)` matrix = ~786K params. The token embedding table alone is roughly a third of the model's total parameter count at the 124M scale.

**Init** — `model.py:201-202`:
```python
elif isinstance(module, nn.Embedding):
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Same std (0.02) as Linear layers, no scaled-init treatment.

**Usage in forward** — `model.py:222-227`:
```python
pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
x = tok_emb + pos_emb                  # broadcast: (B, T, n_embd)
```

A few things worth noticing:

- The positions are just `[0, 1, 2, ..., T-1]` — *the same for every sequence in the batch*. There's no notion of "where in the document we are" beyond position-within-this-context. If you slide your context window forward by one token, every token is now at a position one less than before, and gets a completely different positional embedding. This is part of why absolute learned positions are brittle.
- Token + position are *added*, not concatenated. This is a deliberate choice: keeping both in the same `n_embd`-dimensional space means the attention layers can learn to "use" or "ignore" the positional component however they want. But it also means the two signals share the same dimensions and have to coexist linearly.
- The positional embedding can only handle positions `0..block_size-1`. That's why `block_size` is a hard limit on context length — `wpe` simply has no row for position 1025 if `block_size=1024`. This is the killer limitation of learned absolute embeddings.

## 7. What's modern (post-GPT-2) for this part?

The token embedding table itself is basically unchanged across the modern lineage — a learned lookup table is still how every transformer maps token IDs to vectors. The interesting evolution has been almost entirely on the **positional** side.

### RoPE (Rotary Position Embeddings)

Used by LLaMA, Mistral, Qwen, DeepSeek, GPT-NeoX — basically every serious open model from 2022 onward. The trick: instead of adding a positional vector to the token embedding at the input, you *rotate* the query and key vectors inside each attention head by an angle that depends on the token's position. Higher dimensions rotate slower, lower dimensions rotate faster (like the hands of a clock at different speeds). The math works out so that the dot product `q_m · k_n` only depends on `m - n` — the *relative* position. Consequences:

- No `wpe` parameter at all. You delete it.
- Context length isn't baked into a parameter matrix, so you can extrapolate to longer contexts at inference (with some quality loss). And there are well-studied tricks — NTK-aware scaling, YaRN, position interpolation — to extend the trained context cleanly. This is how models trained at 4k get extended to 32k/128k.
- Relative position is more useful than absolute for language: "the word three tokens back" is a more semantically meaningful concept than "the word at position 847."

### ALiBi (Attention with Linear Biases)

Used by BLOOM, MPT. Even simpler: no positional embedding anywhere; instead, the attention score `q · k` gets a fixed bias `-m * |i - j|` added to it, where `m` is a per-head constant. Closer tokens get penalized less, farther ones more. Trains fast and extrapolates remarkably well, but RoPE has won the popularity contest because it composes better with various attention optimizations.

### NoPE (No Positional Encoding)

Recent research (Kazemnejad et al. 2023) showed decoder-only causal models can sometimes learn positional information *implicitly* through the causal mask alone, with no explicit positional signal at all. The mask makes position-1 fundamentally different from position-7 (different set of things it can attend to), and that asymmetry leaks enough signal for the model to figure out order. Used in some experimental setups; not mainstream yet.

### On the token embedding side

Two minor things worth knowing:

- **Tying is no longer universal**: GPT-2 ties input and output embeddings to save params; LLaMA and most modern models with large vocabs (32k+) *don't* tie, because at large scale the extra capacity in a separate `lm_head` is worth more than the parameter savings.
- **Vocab size has grown a lot**. GPT-2 used 50k BPE tokens; LLaMA-3 uses 128k; modern multilingual tokenizers go to 256k. The embedding table can become the single largest parameter group, which has revived interest in techniques like factorized embeddings (ALBERT-style: decompose the `vocab × d_model` matrix into `vocab × d_small @ d_small × d_model`) — though most production models still just eat the parameter cost.

## Bottom line for this codebase

The modernization that would have the biggest practical impact is swapping learned absolute positions (`wpe`) for RoPE. It's maybe ~50 lines of code, removes a parameter, and unlocks the ability to extend context at inference time. The token embedding table I'd leave alone.
