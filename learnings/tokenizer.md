# Tokenizer

Notes on the tokenizer used in this repo — what BPE is, what `tiktoken` is, where it runs in our code, the critical "gotcha" details (encode_ordinary, leading spaces, EOT), the architectural separation from the model, and a survey of the broader tokenizer landscape.

Companion to [`embedding.md`](./embedding.md): that file covers the *embedding lookup* (token IDs → vectors); this file covers what comes *before* it (raw text → token IDs).

## 1. What is a tokenizer? What is BPE?

A transformer doesn't process text. It processes integers. The tokenizer is the layer that converts between them.

We use **BPE (Byte-Pair Encoding)** — specifically OpenAI's GPT-2 BPE encoding, accessed via the `tiktoken` library. BPE works by:

1. Start with bytes (256 base tokens).
2. Repeatedly find the most frequent pair of adjacent tokens in the training corpus, merge them into a new token. Record the merge.
3. Repeat until you hit the target vocab size (50,257 for GPT-2).

Encoding a new string is then "apply the same merges, in the same order." Deterministic, greedy, easy to implement. **Byte-level BPE** (GPT-2's variant) starts from raw bytes rather than Unicode characters — meaning *any* string can be tokenized losslessly with zero out-of-vocabulary issues, including emoji, weird Unicode, and broken UTF-8.

Examples of what tokenization looks like:

| Text | Tokens | Token IDs |
|---|---|---|
| `"hello world"` | `["hello", " world"]` | `[31373, 995]` |
| `"unbelievable"` | `["un", "bel", "iev", "able"]` | `[403, 6667, 11203, 540]` |
| `"GPT-2"` | `["G", "PT", "-", "2"]` | `[38, 11571, 12, 17]` |
| `"日本"` | (split into UTF-8 bytes) | `[33768, 98, 17312, 105]` |

A single token can be a whole common word (`"hello"`), a word with leading space (`" world"`), a fragment of a rare word (`"un"`, `"bel"`), a single character for rare symbols, or a single UTF-8 byte for languages it wasn't trained on much.

**Why BPE and not chars?** Two reasons:

1. **Sequence length.** English averages ~4 characters per BPE token, so sequences are ~4× shorter than char-level. Attention is `O(T²)`, so 4× shorter sequences are 16× cheaper in attention compute.
2. **Useful units.** "hello" carrying meaning as one indivisible vector is more learnable than the model having to reconstruct it from 5 character vectors.

**Known weaknesses of BPE:**
- Can't easily count letters (`"how many r's in strawberry?"` — the model sees `["str", "aw", "berry"]`, no individual `r`s).
- Inconsistent number tokenization (`"1234"` might be `["12", "34"]` or `["1", "234"]` depending on context). Newer tokenizers (GPT-4, LLaMA) fix this by always splitting digits.
- Non-English text is expensive (falling back to byte-level eats multiple tokens per character).

## 2. `tiktoken` — what it is

`tiktoken` is OpenAI's open-source Rust-backed BPE library, released alongside the ChatGPT API in late 2022. It's roughly 100× faster than the original Python `gpt-2/encoder.py`. Repo: github.com/openai/tiktoken.

It ships several pre-built encodings:

| Encoding name | Vocab | Used by |
|---|---|---|
| `gpt2` (= `r50k_base`) | 50,257 | GPT-2 (our model), GPT-3 |
| `p50k_base` | 50,281 | GPT-3 codex variants |
| `cl100k_base` | 100,277 | GPT-3.5, GPT-4 |
| `o200k_base` | 200,019 | GPT-4o, o1 |

We use `gpt2`. The encoding is a fixed merge table plus a regex pre-tokenizer; tiktoken downloads and caches it under `~/.cache/tiktoken/` on first use.

## 3. When does it run? Offline vs online.

**Tokenization runs once, ahead of time, before the first training step ever happens**, and then never again during training.

```
[once, manually]                       [days of training]                  [each generation]
prep_fineweb_edu.py    ────────►   data/*.bin   ────────►   train.py   ────────►   sample.py
(tokenizes ~10B tokens,             (50 GB of                (zero                  (tokenizes
takes hours on a                    uint16 IDs)              tokenizer              one prompt
many-core machine)                                           calls)                 per run)
```

Why offline:
- Tokenization is **pure CPU work** (regex matching + dict lookups, no math).
- It's **deterministic and reusable** — no reason to redo it every epoch.
- Tokenizing 10B tokens takes hours even with multiprocessing. You do not want that on the critical path of every training run.

Once `.bin` files exist, the GPU never waits on a CPU process — it just reads contiguous `uint16` buffers and reshapes them to `(B, T)`. See `data.py:37`: `np.fromfile(path, dtype=np.uint16)`.

The only downside: if you change your tokenizer mid-project, you regenerate every shard from scratch. In practice you pick a tokenizer once and never touch it.

## 4. Where it lives in our code

Four files. Notably **NOT** in `model.py` or `train.py` — the model is tokenizer-agnostic once data exists in `.bin` form. This separation is deliberate (see §6).

```
scripts/prep_fineweb_edu.py:26   import tiktoken    ← offline, main training data
scripts/prep_shakespeare.py:18   import tiktoken    ← offline, toy/dev dataset
sample.py:19                     import tiktoken    ← online, inference
eval_hellaswag.py:31             import tiktoken    ← online, eval
```

### 4a. Offline — `scripts/prep_fineweb_edu.py`

This is the big one. Produces the `.bin` shards `DataLoaderLite` reads.

**Setup** (`prep_fineweb_edu.py:37-38`):
```python
_enc = tiktoken.get_encoding("gpt2")
_EOT = _enc._special_tokens["<|endoftext|>"]  # 50256
```

**The encode call** (`prep_fineweb_edu.py:41-52`):
```python
def _tokenize(doc: dict) -> np.ndarray:
    text = doc["text"]
    tokens = [_EOT]
    tokens.extend(_enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.uint32)
    assert (tokens_np < 2**16).all(), "Token id out of uint16 range — vocab mismatch."
    return tokens_np.astype(np.uint16)
```

Parallelized with `multiprocessing.Pool` across half the CPU cores (`prep_fineweb_edu.py:75-82`).

### 4b. Offline — `scripts/prep_shakespeare.py`

Same idea, simpler. Single tiny file → `train.bin` + `val.bin`. Useful for sanity-checking the training loop on a CPU/laptop without downloading 10B tokens.

```python
enc = tiktoken.get_encoding("gpt2")               # line 37
tokens = enc.encode_ordinary(text)                # line 38
```

### 4c. Online — `sample.py`

This is the only place you see *both* directions (encode for input, decode for output):

```python
enc = tiktoken.get_encoding("gpt2")                       # line 54
prompt_ids = enc.encode_ordinary(args.prompt)             # line 55 — encode

x = torch.tensor([prompt_ids] * args.n_samples,
                 dtype=torch.long, device=device)         # line 58 — shape (B, T)
out = model.generate(x, max_new_tokens=...)               # line 61 — shape (B, T+new)

for i in range(args.n_samples):
    text = enc.decode(out[i].tolist())                    # line 67 — decode
    print(text)
```

`decode` takes a Python list of ints and returns the reconstructed string. Lossless: `decode(encode_ordinary(s)) == s` for normal text. Note the `.tolist()` — tiktoken wants a plain list, not a torch tensor.

During generation we don't tokenize again at each step — `model.generate` produces token IDs directly from `multinomial(softmax(logits))` (`model.py:321`) and appends them to the running tensor. The tokenizer only runs *once* at the start (encode prompt) and *once* at the end (decode final sequence).

### 4d. Online — `eval_hellaswag.py`

HellaSwag is a multiple-choice commonsense benchmark. For each question we compute the model's likelihood for each candidate ending and pick the highest. Tokenization happens twice per example:

```python
enc = tiktoken.get_encoding("gpt2")                       # line 99
ctx_ids = enc.encode_ordinary(ctx)                        # line 66
end_ids = enc.encode_ordinary(" " + ending)               # line 72
```

The leading space in `" " + ending` is critical — see §5.

## 5. Critical details / gotchas

### `encode_ordinary` vs `encode`

`encode_ordinary` *refuses to interpret special tokens in the input string*. If a malicious dataset contained the literal string `"<|endoftext|>"`, `encode` would happily produce token ID 50256 in the middle of a document — a prompt injection / data poisoning vector. `encode_ordinary` instead tokenizes that string as a sequence of regular tokens.

**Always use `encode_ordinary` for untrusted text.** This codebase does, everywhere.

### Leading-space matters

BPE encodes leading spaces as part of the *following* token. So `"world"` and `" world"` are completely different tokens with different IDs:

```python
enc.encode_ordinary("world")    # [6894]
enc.encode_ordinary(" world")   # [995]
```

When concatenating strings before tokenization (e.g., gluing an ending onto a context in HellaSwag), the space goes *with* the new chunk:

```python
end_ids = enc.encode_ordinary(" " + ending)   # ✓ — leading space attached to ending
end_ids = enc.encode_ordinary(ending)         # ✗ — first token of ending will be wrong
```

Easy mistake to make. Visible in `eval_hellaswag.py:72`.

### `<|endoftext|>` (EOT, token 50256)

GPT-2's vocab has exactly one special token: `<|endoftext|>`, ID 50256. It separates documents. We **prepend** it to each document during prep (`prep_fineweb_edu.py:48`):

```python
tokens = [_EOT]
tokens.extend(_enc.encode_ordinary(text))
```

That way when `DataLoaderLite` stitches docs into a contiguous stream, each doc starts with a clear "fresh start" signal. The model learns that whatever follows EOT is independent of what came before. (Prepending vs appending is a convention choice; nanoGPT picked prepend.)

Accessing it via `_enc._special_tokens["<|endoftext|>"]` is mildly gross — that's tiktoken's API.

### `uint16` cast

GPT-2's vocab is 50,257, fits in 16 bits (max uint16 = 65,535). Saving shards as `uint16` halves disk + I/O cost vs `uint32`. The assert in `_tokenize` guards against silently using a tokenizer with vocab >65,535 (e.g., `cl100k_base` at ~100k — would overflow). The cast back to `int64` happens once when each shard is loaded (`data.py:37-38`), since `nn.Embedding` requires `int64` indices.

## 6. Architectural separation: tokenizer and model are decoupled

```
                      ┌──────────────────────────────┐
                      │  tiktoken("gpt2") — frozen   │
                      └──────────────┬───────────────┘
                                     │ used by
        ┌────────────────────────────┼─────────────────────────┐
        ▼                            ▼                         ▼
   prep_fineweb_edu.py          sample.py                eval_hellaswag.py
   prep_shakespeare.py          (online, prompt)         (online, context+endings)
   (offline, before training)        │                         │
        │                            ▼                         ▼
        ▼                       model.generate              model.forward
   data/*.bin (uint16)               ▲                         ▲
        │                            │                         │
        └──► DataLoaderLite ─────────┴── (B, T) int64 ─────────┘
              (no tokenizer needed)             │
                                                ▼
                                    model.forward(idx, targets)
                                    [ model.py — pure tensors,
                                      tokenizer-free ]
```

Why this matters:
- The model can be served on a machine without `tiktoken` — clients tokenize and send token IDs over the wire (production inference servers work this way).
- Training never blocks on tokenization. CPU work is amortized to a one-time prep step; the hot loop is pure GPU.
- You can swap model architectures without touching the tokenizer code.

The tokenizer and the model are **tightly coupled in meaning but separately built**. The tokenizer is pre-built and frozen for the model's entire life. The model only learns what each token ID *means* (via `wte`); it never has any say in how text gets chunked into IDs. Swap tokenizers → the trained weights become meaningless gibberish (token ID 15496 means `"Hello"` only because `wte` row 15496 was trained with `"Hello"` as input).

## 7. The broader tokenizer landscape

### Algorithmic families

Four main approaches, all balancing the same tradeoff (small vocab = cheap; large vocab = common units stay intact):

**BPE** — what we use. Bottom-up: start from bytes, merge most-frequent pairs greedily. Dominant family today: GPT-2/3/4, LLaMA-1/2/3, Mistral, Qwen, DeepSeek. Variant: **byte-level BPE** (GPT-2's choice) starts from raw bytes instead of Unicode chars → handles any string losslessly.

**WordPiece** — BERT's tokenizer. Same family as BPE, but merges the pair that maximizes corpus likelihood under a unigram LM (BPE picks pairs by raw count; WordPiece picks by information). Largely replaced by BPE in the decoder-only world.

**Unigram LM (Kudo, 2018)** — SentencePiece's default mode, used by T5, mBART, ALBERT. Works *opposite* from BPE: start with a *huge* candidate vocab, iteratively *prune* tokens whose removal hurts likelihood the least. Produces probabilistic tokenization (a string can be tokenized multiple ways with different probabilities, useful for regularization). Multilingual-friendly.

**SentencePiece** — Google's library wrapping both BPE and Unigram. Its main contribution: treats input as raw bytes/Unicode *without language-specific pre-tokenization* (no need for whitespace splitting). Critical for Japanese, Chinese, Thai. LLaMA-1/2 used SentencePiece with BPE inside.

### Production tokenizers — what real models use

| Model | Tokenizer | Vocab | Notes |
|---|---|---|---|
| **GPT-2** (this repo) | tiktoken `gpt2` (byte-level BPE) | 50,257 | English-centric. Spaces attached to following token. |
| **GPT-3** | tiktoken `r50k_base` | 50,257 | Same encoding as GPT-2 |
| **GPT-3.5 / GPT-4** | tiktoken `cl100k_base` | 100,277 | Better multilingual; **digits split into 1–3 char chunks** (helps arithmetic); keeps code structures intact |
| **GPT-4o / o1** | tiktoken `o200k_base` | 200,019 | More multilingual; ~1.4× more efficient than `cl100k_base` for non-English |
| **Claude** | proprietary BPE | ~100k+ | Anthropic doesn't publish; broadly comparable to GPT-4 era |
| **LLaMA 1 / 2** | SentencePiece BPE | 32,000 | Small vocab, Western-language focus, individual digit splitting |
| **LLaMA 3** | tiktoken-style BPE | 128,256 | Major upgrade. Left SentencePiece behind. |
| **Mistral / Mixtral** | SentencePiece BPE | 32k / 32,768 | Inherited from LLaMA family |
| **Qwen** | tiktoken-style BPE | 151,936 | Strong Chinese coverage |
| **DeepSeek** | tiktoken-style BPE | 100,000+ | Tuned for code + multilingual |
| **Gemma** | SentencePiece BPE | 256,000 | Huge vocab for multilingual + code |
| **BERT** | WordPiece | 30,522 | Encoder-only, older era |
| **T5 / mT5** | SentencePiece Unigram | 32k / 250k | Encoder-decoder |

### Patterns to notice

- **Vocab size is climbing**. GPT-2 (50k) → GPT-4 (100k) → GPT-4o (200k) → Gemma (256k). Reason: bigger vocab → fewer tokens per document → longer effective context for the same `block_size` and cheaper inference per character. Cost: bigger embedding table, but at large model scale that's a small fraction of total params.
- **Multilingual coverage drives most of the growth**. GPT-2 falls back to individual UTF-8 bytes for non-Latin scripts (`日本 → [33768, 98, 17312, 105]`). At 100k+ vocab you give Chinese, Cyrillic, Hindi, etc. first-class tokens.
- **Digit handling has been deliberately re-engineered**. Older tokenizers chunked numbers arbitrarily (`"12345"` → some BPE soup), which made arithmetic hard. GPT-3.5+ and LLaMA all split digits more consistently; LLaMA goes all the way and tokenizes every single digit as its own token. Visibly improves math benchmarks. **If a model is bad at arithmetic, look at the tokenizer first.**
- **Most modern models converged on tiktoken-style byte-level BPE**, even when they started with SentencePiece. LLaMA 3 making this switch was notable. Reasons: tiktoken is faster, byte-level handles edge cases (broken Unicode, code, emoji) more cleanly, and interoperates well with existing tooling.

### Modern direction — tokenizer-less / byte-level / learned

Active research line: tokenizers are a hack that introduces a brittle discrete layer between text and the model.

**ByT5 (Google, 2021)** — pure byte-level. Vocab = 256 + a few specials. Perfect robustness to typos, code, multilingual. Cost: sequences ~4× longer, attention is `O(T²)`, so 16× more attention compute. Mostly a research curiosity at scale.

**Charformer (Tay et al., 2021)** — learns tokenization as *part of* the model via Gradient-Based Subword Tokenization (GBST). No hand-engineered BPE step.

**Meta BLT — Byte Latent Transformer (2024)** — most serious recent attempt. Operates on raw bytes but uses *dynamic patching* — a small entropy model groups bytes into variable-length patches (high-entropy regions → fine-grained; low-entropy regions → coarse). Efficiency comparable to BPE but no fixed vocab, much better for low-resource languages. Probably the most likely candidate for the post-BPE world if it pans out at scale.

**Why hasn't tokenizer-less won yet?** Inertia + practical efficiency. BPE has a decade of engineering polish, fits cleanly into the `(B, T, C)` tensor abstraction, and at frontier scale the tokenization weirdness is mostly hidden by enough model capacity. Until a byte-level method demonstrates clearly better quality-per-FLOP at GPT-4 scale, the industry will keep iterating on bigger and better BPE tokenizers. Worth watching: I'd expect at least one frontier model in the next 1–2 years to ship with something more sophisticated than BPE.

## 8. How tokenizers are created

There's a confusingly overloaded word here: "training." A BPE tokenizer is *trained*, but it's a fundamentally different process from training a neural network.

| | Neural net training | BPE tokenizer training |
|---|---|---|
| Input | data + initial random weights | corpus of text |
| Process | gradient descent | greedy pair-merging |
| Hardware | GPU, hours-to-weeks | CPU, minutes-to-hours |
| Output | learned weights | merge table + vocab |
| Stochastic? | yes (init, batching, dropout) | no, fully deterministic |

### The BPE training algorithm, step by step

```
1. Pre-tokenize the corpus using a regex
   "hello world" → ["hello", " world"]
   Each word becomes a sequence of bytes:
     "hello" → [104, 101, 108, 108, 111]   (ASCII codes)

2. Initialize vocab = {0..255}  (every byte is a token)

3. Loop until vocab_size hit:
   a. Count adjacent pairs across the corpus
      e.g., (104, 101) → "he" appears 50,000×
            (108, 108) → "ll" appears 30,000×
   b. Pick the most frequent pair → say (104, 101)
   c. Add it as a new token, say token 256 = "he"
   d. Replace ALL occurrences of (104, 101) in the corpus with 256
   e. Record the merge: "(104, 101) → 256"

4. Output:
   - vocab.json:   id → byte sequence  (256, "he"), (257, "ll"), ...
   - merges.txt:   ordered list of pairs that were merged
```

That's the entire algorithm. To **encode a new string**, apply the same pre-tokenizer regex, then apply the merge list in the same order until no more merges apply. The merge table is "trained" in the sense that it's *derived from data via an algorithm* — but the algorithm is greedy frequency-counting, not gradient descent. Runs on a laptop in an hour for a Wikipedia-sized corpus. No GPU needed, no neural network involved at any point.

### What's human-designed vs data-derived

| Decision | Who picks it |
|---|---|
| **The algorithm** (BPE vs Unigram vs WordPiece) | Human, paper-level decision |
| **Vocab size** (50k vs 100k vs 256k) | Human, hyperparameter |
| **The pre-tokenizer regex** | Human, hand-crafted |
| **Byte-level vs char-level base** | Human |
| **Training corpus selection** | Human |
| **Special tokens to reserve** | Human (see §9) |
| **The merge order and resulting vocab** | Data, via the algorithm |

The pre-tokenizer regex deserves a callout — it's the most underappreciated piece of human engineering in the whole pipeline. **GPT-2's pre-tokenizer regex** (this is a real string from OpenAI's code):

```python
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

Piece by piece:
- `'s|'t|'re|'ve|'m|'ll|'d` — keep English contractions intact (`"don't"` stays as `["don", "'t"]`, not `["do", "n't"]`)
- ` ?\p{L}+` — runs of letters, with optional leading space (this is why `" world"` is one chunk)
- ` ?\p{N}+` — runs of digits, with optional leading space
- ` ?[^\s\p{L}\p{N}]+` — runs of punctuation
- `\s+(?!\S)|\s+` — runs of whitespace

The regex's job: **define atomic units before BPE starts merging.** BPE will then merge bytes *within* these chunks but never *across* them. Without this, BPE might learn weird merges like `"e the"` because "e " and "the" co-occur a lot — which would be useless. The regex prevents that by forcing merges to stay within natural linguistic boundaries.

GPT-4's pre-tokenizer regex is much more elaborate (handles case, contractions in more languages, code-aware splitting). Tokenizer regex engineering is a quiet but consequential craft. Every major lab has their own.

### What corpus is used to train BPE?

The BPE training corpus determines **which strings get compressed into single tokens**. This is the most important data choice in the whole tokenizer pipeline.

| Tokenizer | BPE training corpus | Effect |
|---|---|---|
| **GPT-2 `gpt2`** (this repo) | **WebText** — ~8M docs / ~40 GB from Reddit-linked URLs (3+ karma). Same as GPT-2's model training data. | Almost entirely English. Other languages fall back to byte-level (`日本 → 4 tokens`). |
| **GPT-3 `r50k_base`** | Inherited from GPT-2, not retrained | Same limitations as GPT-2 |
| **GPT-3.5/4 `cl100k_base`** | OpenAI didn't publish — believed to include CommonCrawl + much more code + some multilingual | Better digits, better code, modest multilingual gain |
| **GPT-4o `o200k_base`** | Even more multilingual emphasis | Much better non-English compression |
| **LLaMA-1/2** | Sample of their pretraining mix (CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange) | English-strong, decent code, weak non-Western languages |
| **LLaMA-3** | Multilingual-heavy sample — Meta said "8× more non-English" than LLaMA-2's tokenizer corpus | First-class tokens for many more languages |
| **Gemma** | Heavily multilingual + code | 256k vocab gives room for many languages |
| **Qwen** | Chinese-heavy alongside English | Strong Chinese token coverage |

Two structural points:

1. **BPE training corpus ≠ model training corpus** (necessarily). The BPE training corpus is typically a *sample* — 10s of GB is enough to get reliable pair statistics. The model is then trained on TBs. They don't have to be the same data, though they're usually drawn from the same distribution to avoid mismatch.
2. **You can't fix the tokenizer after the fact.** If your tokenizer was trained on English-only data and you later want your model to speak Korean, you're stuck: Korean characters will eat 2-3 byte-level tokens each, making Korean training ~3× more expensive and Korean inference ~3× slower than necessary. The only fix is training a new tokenizer (which then forces a model retrain). This is one reason LLaMA-3 was a fresh tokenizer, not a fine-tune of LLaMA-2.

### Tools for actually training a BPE tokenizer

You'd never write the algorithm yourself. The standard options:

- **HuggingFace `tokenizers`** — Rust-backed, supports BPE/Unigram/WordPiece, very fast. The pragmatic choice for most projects.
- **`tiktoken`** — OpenAI's library. Has a less-publicized `train_bpe` function. We use the pre-trained one, but it can train new ones.
- **SentencePiece** — Google's C++ library. The choice if you need Unigram or want LLaMA-style tokenizers.
- **Karpathy's `minbpe`** — pure-Python educational implementation, ~400 lines. **Highly recommend reading this if you want to actually understand BPE.** Repo: github.com/karpathy/minbpe

### For our codebase

We didn't train our own tokenizer — `tiktoken.get_encoding("gpt2")` downloads OpenAI's pre-trained merge table from their CDN. The training was done by OpenAI in 2018 on WebText. Once you have the merge table file, encoding is just "apply these merges" — fully reproducible, no randomness.

## 9. Special tokens — how they're added

Special tokens are **manually reserved IDs** that:

1. Are **never produced by the BPE algorithm** — they live outside the BPE process.
2. Have **structural meaning**, not textual meaning — they're "control codes" the model uses to organize input.
3. Get **their own embedding row** that the model learns the meaning of during pretraining or fine-tuning.

### Two ways they get into a tokenizer

**(a) Reserved at vocab-build time** — the most common. The tokenizer designer decides "ID 50256 is `<|endoftext|>`" *before* training the model. BPE only produces IDs 0–50255, and the model's `vocab_size` is set to include the reserved slot.

This is what GPT-2 does. The vocab is exactly 50,257 = 50,256 BPE tokens + 1 special token (EOT at ID 50256).

```python
_enc = tiktoken.get_encoding("gpt2")
_enc._special_tokens
# {'<|endoftext|>': 50256}
```

The mechanism that prevents accidents:
- **The BPE merge algorithm never sees special tokens** because they're either stripped from the corpus before BPE training, or (more commonly) inserted *after* tokenization — which is exactly what we do in `prep_fineweb_edu.py:48`: `tokens = [_EOT] + enc.encode_ordinary(text)`.
- **`encode_ordinary` refuses to interpret them** in user text. If your prompt literally contains the string `"<|endoftext|>"`, `encode_ordinary` tokenizes the characters `<`, `|`, `e`, ... as normal text. Only `encode(..., allowed_special={"<|endoftext|>"})` will produce ID 50256 from user input. This is exactly the prompt-injection defense from §5.

**(b) Added post-hoc** — sometimes you want to add tokens *after* the model is pretrained. E.g., fine-tuning LLaMA-2 (no chat tokens) into a chat model. You extend the tokenizer's vocab and also extend the embedding table (`wte.weight` gains new rows initialized randomly, and `lm_head` matches — they're tied). Then you fine-tune so the new rows learn meaningful representations.

This is more fragile because the new embedding rows start from random init in an already-trained network. Common practice is to initialize them to the mean of existing embeddings rather than random, to reduce the initial perturbation.

### Concrete examples — what specific models reserve

**GPT-2** — minimal, single token:
- `<|endoftext|>` (50256) — document separator + end-of-generation signal

**GPT-3.5/4 (`cl100k_base`)** — small chat-era additions:
- `<|endoftext|>` (100257)
- `<|fim_prefix|>` (100258), `<|fim_middle|>` (100259), `<|fim_suffix|>` (100260) — fill-in-the-middle for code
- `<|endofprompt|>` (100276)
- `<|im_start|>` / `<|im_end|>` — added later for ChatML format

**LLaMA-3** — went all-in on reserving for the future:
- `<|begin_of_text|>` — document start (LLaMA convention; reversed from GPT-2's prepend pattern)
- `<|end_of_text|>` — document end
- `<|start_header_id|>`, `<|end_header_id|>` — wrap role names like `user`, `assistant`, `system`
- `<|eot_id|>` — end of a chat turn
- `<|python_tag|>`, `<|eom_id|>` — tool calling
- **256 reserved slots for future use** (`<|reserved_special_token_0|>` through `<|reserved_special_token_250|>`). Meta literally said: "we don't know what we'll need, but we want the option."

**OpenAI o1 / o3 ("thinking" models)** — uses special tokens to delimit chain-of-thought reasoning that's hidden from the user. Almost certainly something like `<|thinking_start|>` and `<|thinking_end|>` (exact strings are proprietary). The model is trained to put internal reasoning between these tags; the API strips them before returning to the user. The token IDs sit in the high range of `o200k_base` (200,019 vocab gives plenty of room).

**Anthropic Claude** — uses XML-like control structures (`<thinking>`, `<answer>`, etc.). Some of these are likely special tokens internally, but Anthropic doesn't publish details.

**Code models** (StarCoder, DeepSeek-Coder, Codex) — typically reserve:
- `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>` for fill-in-the-middle training
- `<file_sep>` for separating files in repository context
- `<|repo_name|>`, `<|file_name|>` for metadata

### What the model has to "know" about a special token

A special token starts as just an embedding row. The *meaning* of that row — "this token signals end-of-document, so I should not predict tokens from the previous doc as if they're context" — is learned entirely from training data.

Concretely, for `<|endoftext|>` to mean what we want it to mean, the training data must contain millions of document boundaries with EOT inserted between them. Then:

1. The model sees `... last token of doc A. <|endoftext|> First token of doc B ...`
2. It learns that "after EOT, the previous context is no longer informative."
3. The embedding row 50256 develops the right structure to communicate this to subsequent layers.

If you add a new special token and only fine-tune on a handful of examples, the model will barely have any signal to learn what it means. That's why chat fine-tuning datasets are typically tens of thousands to millions of examples.

### Chat templates — the layer above special tokens

Modern chat models layer a **chat template** on top of the special tokens. This is a Jinja-style template that converts a structured conversation:

```python
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is 2+2?"},
]
```

…into a flat string with special tokens interleaved (LLaMA-3 example):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

This string then gets tokenized normally. The chat template is *not* part of the tokenizer or the model — it lives in the model card as a config string, and HuggingFace's `apply_chat_template` runs it. Same model can be served with different chat templates by different downstream wrappers (which is why you sometimes see chat models behave oddly — wrong template applied).

For our GPT-2 reproduction we don't have any of this — there's no chat, no system prompt, no tool calling. Just `<|endoftext|>` for document boundaries.

## 10. Practical recommendation for this codebase

We're reproducing GPT-2 faithfully, so `tiktoken("gpt2")` is the right choice — no reason to swap.

If you ever want to experiment, the easiest swap that would teach you something is replacing `get_encoding("gpt2")` with `get_encoding("cl100k_base")` and re-prepping the data. You'd see:
- Smaller `.bin` files (fewer tokens per document).
- Need to update `vocab_size = 100277` (and pad to multiple of 64 → `100352`) in `GPTConfig`.
- Need to change the uint16 cast in `prep_fineweb_edu.py` to uint32 (100k > 65535).
- The model would handle multilingual prompts much better.
- You'd need to retrain from scratch.

Not worth doing for its own sake, but a clean exercise if you want to feel how the tokenizer ripples through the rest of the system.
