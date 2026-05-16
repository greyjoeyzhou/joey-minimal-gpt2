# Training Data

Notes on the data used to pretrain this model — what's in the repo, the lineage from CommonCrawl → FineWeb → FineWeb-Edu → our sample, the broader landscape of open pretraining datasets, and the always-underestimated problem of duplication.

Companion to [`tokenizer.md`](./tokenizer.md) (which covers how text becomes tokens) and [`embedding.md`](./embedding.md) (which covers how tokens become vectors).

## 1. What's in this repo

Two datasets are wired up, with very different roles:

**Tiny Shakespeare — the toy/dev dataset** (`data/shakespeare/`)
- 1.1 MB of raw text (~300K tokens) — the complete works of Shakespeare, concatenated.
- Prepared by `scripts/prep_shakespeare.py`.
- Sole purpose: sanity-check that the training loop runs end-to-end. Fits in memory on a laptop CPU; can be overfit in minutes. If `train.py` doesn't reduce loss on Shakespeare, something is fundamentally broken.
- **Not** what the real model trains on.

**FineWeb-Edu 10B sample — the real training data** (`data/edu_fineweb10B/`)
- 10 **billion** tokens of educational web text. ~20 GB on disk as uint16 token IDs.
- Prepared by `scripts/prep_fineweb_edu.py`.
- This is what `train.py`'s 19,073-step run targets: 19,073 steps × 524,288 tokens/step ≈ 10B tokens ≈ exactly one epoch.

The directory is empty on the Mac (prep is meant to run on the 5090 box). On a many-core machine, prep takes a few hours; on a 5090, training takes ~24 hours.

## 2. The data lineage

```
CommonCrawl
   raw web crawls, monthly dumps since 2008
   ~400 TB per monthly dump (WARC files), cumulative tens of PB
        │
        │  [HuggingFace processing — June 2024]
        │  URL filtering, language ID, quality heuristics,
        │  MinHash dedup at scale
        ▼
FineWeb
   15 trillion tokens (~44 TB of clean text)
   95 CommonCrawl dumps from 2013–2024
        │
        │  [educational-quality classifier filter]
        │  Llama-3-70B labels pages, small classifier scores them,
        │  keep only high-scoring pages
        ▼
FineWeb-Edu
   1.3 trillion tokens (~3.5 TB)
        │
        │  [random sample]
        ▼
sample-10BT  ← THIS is what we train on
   10 billion tokens (~20 GB on disk as uint16)
   ~0.77% of FineWeb-Edu, ~0.067% of FineWeb
```

### CommonCrawl — the foundation

A non-profit web archive that has been crawling the web continuously since 2008. They release monthly "dumps" of crawled HTML to a public S3 bucket — **free to download, no auth required**. Each monthly dump is ~300–400 TB of raw WARC files (a web-archive format that includes HTTP headers, HTML, sometimes JavaScript, basically whatever they got).

This is the **single largest source of pretraining data on the planet** and underlies essentially every major LLM dataset (and almost certainly every frontier model's training mix, though closed labs don't publish details). It would cost tens of millions of dollars to replicate, and nobody has bothered.

But CommonCrawl is **raw and dirty**:
- Massive duplication (the same article reposted across 100 sites, navigation boilerplate on every page)
- Tons of non-content (cookie banners, footers, JavaScript, ad markup)
- Spam, SEO garbage, machine-translated text, low-quality content
- ~50%+ non-English (depends on the dump)
- Pages re-crawled across many dumps

Training on raw CommonCrawl produces a mediocre model. Every serious lab does heavy processing on top.

### FineWeb — HuggingFace's processed CommonCrawl

**FineWeb** (Penedo et al., June 2024) is HuggingFace's open-source attempt to produce a clean, deduplicated, filtered version of CommonCrawl that matches what closed labs use internally. Paper: "FineWeb: decanting the web for the finest text data at scale."

- **Source**: 95 CommonCrawl dumps spanning 2013–2024
- **Final size**: **15 trillion tokens** (~44 TB of text)
- One of the largest **fully open** pretraining datasets ever released.

The pipeline (each step ablated in the paper, with measured impact on downstream benchmarks):

1. **URL filtering** — block known low-quality / adult / spam domains using a blocklist.
2. **Language identification** — keep only English (using fastText).
3. **Quality filters** — drop pages that fail heuristics like "ratio of stop words to total words", "average line length", "fraction of lines ending in punctuation". These are the **C4 filters** (from the T5 paper) + **Gopher quality filters** (from DeepMind's Gopher paper).
4. **Repetition filtering** — drop pages with too much within-document repetition.
5. **Deduplication** — MinHash LSH at scale. See §4 for why this is the most important step.

The reason FineWeb matters as a *reference point* is that the paper publishes detailed ablations: train a small model on each variant, measure HellaSwag/ARC/MMLU/etc., compare. We have empirical evidence for what each filtering step is worth, instead of vibes.

### FineWeb-Edu — the educational subset

A subset of FineWeb produced by a single additional filtering step: an **educational-quality classifier**.

- **Source**: full FineWeb
- **Final size**: **1.3 trillion tokens** (~3.5 TB) — about 8.5% of FineWeb survives
- Released in the same FineWeb paper

How the classifier was built (this pattern is now standard):

1. Sample 500K web pages from FineWeb
2. Use **Llama-3-70B** to label each on a 0–5 scale for "educational value" (does this look like content from a textbook, lecture, scientific article, etc.?)
3. Train a **small classifier** (Snowflake's `arctic-embed-m` embeddings + a linear head) on those labels
4. Run the classifier across all of FineWeb, keep pages scoring ≥3

The bootstrap pattern: **use a big expensive model to label, distill into a cheap classifier, scale the classifier across the whole corpus.**

Key empirical finding: a model trained on FineWeb-Edu **beats** a model trained on full FineWeb at the same compute budget, on academic-style benchmarks (HellaSwag, MMLU, ARC). The "quality > quantity" story playing out at scale, consistent with Microsoft's earlier Phi-1/2/3 findings.

Flip side: FineWeb-Edu is biased *toward* academic-style writing and *away* from conversational text, fiction, social media. A chatbot fine-tuned from a FineWeb-Edu base might sound stiff. For benchmark-chasing this is fine; for general-purpose models you'd want to mix.

### sample-10BT — what we use

HuggingFace publishes smaller random subsets of FineWeb-Edu for experimentation:

- `sample-10BT` — 10B tokens (~20 GB) ← what we use
- `sample-100BT` — 100B tokens
- `sample-350BT` — 350B tokens
- Plus the full 1.3T-token version

Our `prep_fineweb_edu.py:31-32` selects this:
```python
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
```

The math behind picking 10B:

- Our model: GPT-2 124M parameters.
- "Chinchilla-optimal" tokens (DeepMind's 2022 scaling law): ~20 tokens per parameter → ~2.5B tokens.
- **We're using 4× Chinchilla-optimal** (10B / 2.5B), the modern "overtrained small model" regime — same philosophy as LLaMA-1 (7B model on 1T tokens, ~140× Chinchilla).
- Why overtrain a small model? **Inference cost.** A 124M model overtrained on 10B tokens is cheaper to serve forever than a 7B model trained on 2.5B tokens, and quality is often comparable. Chinchilla optimized for training cost; in production you mostly care about inference cost.
- We don't go bigger than 10B because: (a) on a single 5090, 10B / 524K tokens-per-step = 19,073 steps ≈ 24 hours. 100B would mean ~10 days. (b) Returns diminish.

### Layer-by-layer size summary

| Layer | Size | What we use | % |
|---|---|---|---|
| CommonCrawl (cumulative) | ~tens of PB raw | 10 GB worth of processed text | rounding error |
| FineWeb | 15T tokens | 10B tokens | 0.067% |
| FineWeb-Edu | 1.3T tokens | 10B tokens | 0.77% |
| sample-10BT (random subset) | 10B tokens | 10B tokens | 100% |

Almost all the heavy lifting was done by HuggingFace's processing pipeline. We're consuming a tiny, conveniently-packaged slice of their output.

## 3. The broader landscape of pretraining datasets

A working taxonomy:

```
PRETRAINING DATA
├── Web crawl + filtering (general)         [FineWeb, DCLM, RefinedWeb, C4]
├── Conversational / forum                  [Reddit, StackExchange, HN]
├── Encyclopedic / reference                [Wikipedia, Wiktionary]
├── Books (long-form prose)                 [Project Gutenberg, Books3*]
├── Specialized domains
│   ├── Code                                [The Stack v2, GitHub]
│   ├── Math                                [OpenWebMath, ProofPile-2]
│   └── Scientific                          [ArXiv, PubMed Central, S2ORC]
└── Synthetic                               [Phi data*, Cosmopedia, Nemotron]

POST-TRAINING DATA (different stage — not the focus here)
├── Instruction tuning (SFT)                [Alpaca, ShareGPT, OpenAssistant]
├── Preference (RLHF/DPO)                   [UltraFeedback, Anthropic HH]
└── Tool use / agentic                      [ToolBench, etc.]

(* = legally fraught / not openly released)
```

**A few framing points before the list:**

- **Almost everything ultimately traces back to CommonCrawl.** The diversity comes from *how* it was filtered/deduplicated/mixed, not from independent sources.
- **The trend over time has been**: small + curated (Wikipedia, Books) → big + raw (Common Crawl dumps) → bigger + heavily curated (FineWeb, DCLM) → model-curated (let an LLM label what's "good"). We're firmly in the model-curated era now.
- **Specialization is rising**: pure-code datasets, pure-math datasets, pure-educational datasets. Modern training mixes blend them in measured ratios.
- **Cross-cutting axis: language coverage.** GPT-2 is ~99% English; LLaMA-3 is ~25% non-English. Doesn't fit neatly into the taxonomy — it cross-cuts every category.

### General-purpose web datasets

| Dataset | Size | By | Year | Notable |
|---|---|---|---|---|
| **WebText** | ~40 GB / ~10B tokens | OpenAI | 2019 | What GPT-2 trained on. **Never released** (closed). Reddit-linked URLs with 3+ karma. |
| **C4** (Colossal Clean Crawled Corpus) | ~750 GB / ~150B tokens | Google | 2019 | First widely-used cleaned CommonCrawl. Heuristic filters (the "C4 filters" everyone still cites). Used to train T5. |
| **The Pile** | 825 GB / ~300B tokens | EleutherAI | 2020 | The dataset that *started the open-LLM movement*. Mix of 22 sources: CC + ArXiv + PubMed + GitHub + Books3 + StackExchange + Wikipedia. Trained GPT-J, GPT-NeoX, Pythia. Books3 portion later removed due to copyright takedown. |
| **RefinedWeb** | 5T tokens | TII (UAE) | 2023 | Behind the Falcon models. **Aggressive deduplication** was the headline contribution. Only ~600B released publicly. |
| **RedPajama-v1** | 1.2T tokens | Together AI | 2023 | Open replication of LLaMA-1's training mix. |
| **RedPajama-v2** | **30T tokens** | Together AI | 2023 | Massive 100+ CommonCrawl dumps + quality signals pre-computed. **Largest open web dataset by raw token count.** |
| **SlimPajama** | 627B tokens | Cerebras | 2023 | RedPajama-v1 with extra deduplication. |
| **Dolma** | 3T tokens | AI2 | 2024 | Behind the fully-open OLMo models. Notable for **full reproducibility** — everything (data, code, weights) is open. |
| **FineWeb** | **15T tokens** | HuggingFace | 2024 | Currently the **highest-quality large open CC processing**. |
| **FineWeb-Edu** | 1.3T tokens | HuggingFace | 2024 | Our dataset. Educational filter on top of FineWeb. |
| **DCLM** (DataComp-LM) | 4T tokens (filtered from 240T pool) | Apple + UW | 2024 | Released alongside a *benchmark* — see which filtering wins. Best-performing fully-open dataset per token. Used in Apple's MM1. |
| **Nemotron-CC** | 6T tokens | NVIDIA | 2024 | NVIDIA's CC re-processing, with synthetic data augmentation. |

If you wanted to train a "general" model today, the strongest open choices are **DCLM** or **FineWeb-Edu** (depending on whether you want broad coverage or academic-leaning). RedPajama-v2 for raw scale.

### Code

| Dataset | Size | By | Year | Notes |
|---|---|---|---|---|
| **GitHub Code (BigQuery)** | ~3 TB | Google | 2019 | The original CC-of-code. Used by early code models. |
| **The Stack** | 6 TB | BigCode | 2022 | Permissively-licensed GitHub repos only (Apache, MIT, BSD). Addresses copyright concerns. Source for StarCoder. |
| **The Stack v2** | **67 TB** / ~3T tokens | BigCode | 2024 | Built from Software Heritage archive. Source for StarCoder 2. **Largest open code corpus.** |

For models that need to be good at code, mix in ~5-15% code data with general web text. Pure-code models train on much higher fractions.

### Math

| Dataset | Size | By | Year | Notes |
|---|---|---|---|---|
| **OpenWebMath** | 14B tokens | Together / various | 2023 | High-quality math from CC (forum posts, blogs, lecture notes). |
| **ProofPile-2** | 55B tokens | EleutherAI | 2023 | Math + scientific papers + theorem-proving data. |
| **MathPile** | 9.5B tokens | Shanghai AI Lab | 2023 | Mix of textbooks, papers, ProofWiki, StackExchange Math. |
| **AlgebraicStack** | 11B tokens | EleutherAI | 2023 | Code that does math (Lean, Coq, Isabelle proofs). |

Improvements measurable on GSM8K and MATH benchmarks.

### Scientific / academic

| Dataset | Size | Notes |
|---|---|---|
| **ArXiv** (LaTeX source) | ~50B tokens | Full ArXiv archive in LaTeX. Dense technical prose with equations. |
| **PubMed Central** | ~10B tokens | Open biomedical papers. |
| **S2ORC** (Semantic Scholar) | ~80M papers | Multi-disciplinary scientific corpus. |

The "high signal-density" sources. Heavily used by every serious training mix.

### Books

**Legally fraught** but practically important — books are uniquely high-quality long-form prose. Models trained without them tend to be worse at long-form generation.

| Dataset | Size | Notes |
|---|---|---|
| **Books3** | ~196k books | Originally part of The Pile. **Taken down 2023** after copyright lawsuit (Sarah Silverman et al. v. OpenAI/Meta). |
| **Project Gutenberg** | ~30k books | Public domain, safe. ~3B tokens. |
| **PG-19** | 28k books | Gutenberg subset published before 1919 (PD). Used as long-document benchmark. |

OpenAI/Anthropic/Google almost certainly use larger book collections internally. **This is the single biggest gap between open and closed training data today.**

### Conversational

| Dataset | Size | Notes |
|---|---|---|
| **Pushshift Reddit** | ~1.5B comments | Reddit dumps. Used by WebText (filtered to high-karma) and many others. Pushshift has been periodically restricted. |
| **StackExchange dumps** | ~10B tokens | Programming Q&A — clean, well-formatted. |
| **HackerNews dumps** | small | Tech discussion. |

*Format-specialized* rather than *topic-specialized*: it's not about subject matter, it's about register (Q&A, dialogue, threaded discussion). Critical for teaching a base model the conversational register.

### Encyclopedic / reference

| Dataset | Size | Notes |
|---|---|---|
| **Wikipedia** | ~20 GB (English) | The canonical example. Disproportionately high signal density. **Every** major mix includes it, often upweighted. |
| **Wiktionary, Wikinews, etc.** | smaller | Related sister projects. |

Small in raw size, but per-token quality is so high it's worth upweighting (sampling multiple times per epoch).

### Synthetic — the rising category

| Dataset | Size | By | Year | Notes |
|---|---|---|---|---|
| **Phi training data** | ~7B tokens | Microsoft | 2023-2024 | Behind Phi-1/2/3/4. Heavily synthetic — GPT-4 generates textbook-style content. **Not released**. Phi papers' headline: small models match much larger ones when trained on this. |
| **Cosmopedia** | 25B tokens / 30M docs | HuggingFace | 2024 | **Open synthetic alternative** to Phi. Mixtral-generated educational content. |
| **Nemotron-4 synthetic data** | ~15B tokens | NVIDIA | 2024 | Synthetic data from Nemotron-4 340B. |

Controversial — active debate about whether synthetic-heavy training causes "model collapse" or memorization of generator biases. But Phi's benchmark wins are real, and everyone is experimenting.

### What a real modern training mix looks like

LLaMA-3 (publicly disclosed) is illustrative:

| Source | Share |
|---|---|
| Web (heavily curated, CC-based) | ~50% |
| Code | ~17% |
| Math + reasoning | ~8% |
| Multilingual | ~25% (much higher than LLaMA-2) |

Frontier closed-model mixes (GPT-4, Claude) are presumed to follow similar shapes — heavy web base, large code fraction, substantial math + scientific content, growing multilingual. Plus large book corpora and synthetic data of unknown quantity.

## 4. Duplication and deduplication

One of the biggest underestimated problems. Duplication shows up in several distinct forms.

### Where duplication comes from

**Within-source duplication** (same dataset, same content multiple times):
- The same article reposted on news syndication sites → CommonCrawl picks up all copies.
- Boilerplate everywhere (footers, cookie banners, "related articles" sections on every page of a site).
- Content scrapers re-hosting blog posts.
- CommonCrawl re-crawls the same sites across monthly dumps — the same Wikipedia article may exist in 50+ dump snapshots.
- Pagination duplicates.

**Cross-source duplication** (different datasets, overlapping content):
- **Wikipedia is everywhere.** Directly in CommonCrawl (wikipedia.org pages), in hundreds of mirrors, quoted in blog posts, embedded in search results. So FineWeb-Edu already contains Wikipedia multiple times, from multiple sources.
- **ArXiv papers** appear in CommonCrawl (academic blogs quote them), in The Pile's CC subset, in OpenWebMath, *and* in a separate ArXiv-LaTeX dataset.
- **Code from GitHub** appears in The Stack, in CommonCrawl (READMEs, gists, code in tutorials), and in Stack Overflow answers.
- The classic case from The Pile (2020): its CommonCrawl component contained pirated copies of books that were *also* in its Books3 component. The same paragraph from Harry Potter could appear 3-4 times.

**Cross-mix duplication** (when you blend datasets):
- Train on RedPajama? That's CC + C4 + Wikipedia + ArXiv + Books. But C4 is filtered CC. And CC contains Wikipedia + ArXiv + Books. So you're "training on" each of these 2-3× without intending to.
- Modern mixes increasingly publish **"effective duplication factors"** — how many times, in expectation, the model sees each piece of unique content.

### How dedup works at scale: MinHash + LSH

The naive "compare every pair of N documents" is `O(N²)` — impossible at 15T tokens. **MinHash + LSH** (locality-sensitive hashing) gives an approximate solution:

1. For each document, compute a short fingerprint (the MinHash signature, ~128 ints).
2. Use LSH to bucket documents likely to be similar into candidate groups.
3. Within each bucket, compute actual Jaccard similarity (n-gram overlap).
4. Documents above ~0.8 similarity → considered duplicates → drop one.

What this catches:
- **Exact duplicates** (byte-for-byte) ✓
- **Near-exact** (minor formatting differences, typos) ✓
- **Paraphrases** (same meaning, different words) ✗ — needs embedding similarity
- **Semantic duplicates** (same facts, different forms) ✗ — research frontier

FineWeb, RefinedWeb, DCLM, Dolma all use MinHash LSH variants.

### The counterintuitive FineWeb finding

The FineWeb paper ran a clean experiment: train identical small models on (a) dedup-per-dump only, (b) dedup-globally-across-all-dumps, (c) raw. Result:

- Global dedup performed **worse** than per-dump dedup, even though it removes more data.
- Why? Content that survives across many dumps is *more likely to be high-quality, evergreen content* (Wikipedia, established sites). Per-dump dedup preserves the natural repetition across time, acting as implicit quality weighting. Global dedup kills this signal — it keeps exactly one copy regardless of how universally referenced.

**Deduplication isn't always strictly better.** It's a knob, and "more aggressive = better" is wrong above a certain point.

### Why duplication matters — three distinct problems

**(a) Wasted compute.** If your training data is 30% duplicates, you're paying 30% extra for no learning gain.

**(b) Memorization.** Lee et al. ("Deduplicating Training Data Makes Language Models Better", 2021): models trained on duplicated data **memorize verbatim** at much higher rates. They can reproduce training examples word-for-word. This is bad for:
- **Privacy**: leaks PII (names, addresses, emails that appeared multiple times)
- **Copyright**: regurgitates copyrighted text verbatim
- **Generalization**: model overfits to memorized passages

Crucially, the relationship is **nonlinear** — seeing something 100× is much more than 100× worse than seeing it once. Small fractions of high-repetition content are dangerous.

**(c) Test-set contamination.** Training data containing evaluation benchmarks:
- HellaSwag questions exist as web text in CC (people quote them in benchmark blog posts).
- GSM8K problems get discussed on forums.
- MMLU questions sometimes appear in study guides.

When the model has "seen" the test set during training, benchmark scores look artificially good. **The single most common source of misleading benchmark numbers in published papers.** Modern dataset pipelines try to remove benchmark content — FineWeb has a "decontamination" step. But it's cat-and-mouse: new benchmarks come out, datasets predate them.

Whether commercial labs do this rigorously is impossible to verify externally.

### Current trends

1. **Smarter deduplication** — moving from exact-match → MinHash near-dup → embedding-based semantic dedup. Dedup compute is becoming non-trivial.
2. **Repetition as a feature, not a bug** — the FineWeb cross-dump finding, plus more recent work showing *some* repetition (1–4×) actually helps generalization. The optimal isn't zero duplication; it's some carefully chosen non-zero amount.

"No duplicates" is the naive ideal; "smart deduplication preserving valuable repetition" is the modern best practice.

## 5. What this means for our setup

We use FineWeb-Edu `sample-10BT`, inheriting HuggingFace's deduplication pipeline:

- Per-CC-dump MinHash LSH dedup ✓
- URL filtering against blocklists ✓
- Some decontamination against common benchmarks (including HellaSwag, which is what `eval_hellaswag.py` evaluates on!) ✓

So our 10B tokens are, in expectation, ~10B *unique-ish* tokens. Not strictly zero duplication, but the worst offenders are gone. We don't need to do anything ourselves.

**If you wanted to experiment beyond this**, the easiest swaps that would teach you something:

- **`HuggingFaceFW/fineweb` (full FineWeb)** — see quality vs. quantity tradeoff first-hand. Same prep script, different config name.
- **`bigcode/the-stack-v2`** — repurpose `prep_fineweb_edu.py` to point at code. Feel what training on pure code looks like.
- **`HuggingFaceTB/cosmopedia`** — try training on synthetic data. Smaller scale, interesting comparison.
- **Mix datasets** — combine FineWeb-Edu + Wikipedia (upweighted) + StackExchange. But beware: each component already contains content from the others, so you reintroduce duplication. Proper mixing requires dedup across the mix.

The moment you start mixing datasets, you reinherit all the duplication problems FineWeb already solved. Trust the pipeline, or rebuild it.
