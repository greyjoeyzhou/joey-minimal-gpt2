# Modern Trends — What's Actually Driving Architectural Change

Meta-notes on *why* post-GPT-2 architectural variants exist. A common framing is "modern variants make training/inference faster without sacrificing quality" — that's partially right, but it flattens three distinct categories and misses the historical context. This doc unpacks what's actually being optimized in each era and which changes are pure wins vs trade-offs.

See also: [`attention-variants.md`](./attention-variants.md), [`transformer-block.md`](./transformer-block.md), [`training-data.md`](./training-data.md).

## The four forces

Architectural changes aren't all about "speed at fixed quality." There are four distinct pressures:

1. **Training compute** efficiency — make the training loop cheaper.
2. **Inference compute and memory** efficiency — make serving cheaper, especially KV cache at long context.
3. **Capability** improvements — better quality at the same cost, or new capabilities (longer context, multimodal).
4. **Operational** improvements — simpler designs, easier to scale, easier to deploy.

Most discussions conflate (1) and (2) under "efficiency," and treat (3) as if it were a tradeoff with efficiency. It usually isn't — many "modern" changes are pure Pareto-frontier improvements.

## What each change is actually optimizing

| Change | What it improves | What it costs |
|---|---|---|
| **Flash Attention** | Training & inference speed, memory | Nothing — identical math |
| **RMSNorm** | Tiny training speed | Nothing — comparable quality |
| **Removing biases** | Tiny everything | Nothing measurable |
| **Tied embeddings** (we have this) | Params (saves ~38M at 124M scale) | Maybe a tiny quality bump |
| **GQA / MQA / MLA** | **Inference** KV cache memory | Small quality loss (varies) |
| **Sliding window attention** | Memory at long context | Quality at long range |
| **Linear / sparse attention** | Theoretical efficiency | Real quality loss (mostly abandoned) |
| **RoPE** | **Context length extension** at inference | Nothing — comparable quality at trained range |
| **SwiGLU** | **Quality** at same param count | Marginally more compute |
| **Better data** (FineWeb-Edu, DCLM, synthetic) | **Quality** | Data prep compute |
| **MoE** (Mixtral, DeepSeek-V3) | **More capacity per inference cost** | More total params, more training compute |
| **Hybrid SSM** (Jamba) | Long-context memory | Recall quality (somewhat) |

## Three distinct patterns

### (a) Pure wins — zero quality cost, real efficiency gain

**Flash Attention. RMSNorm. Removing biases.**

These are the "everyone should do this" changes. If you're not using them, it's just laziness or backwards-compatibility. They are not tradeoffs — they're free upgrades.

### (b) Inference-cost trades — small quality loss for big serving savings

**GQA, MQA, MLA, SWA.**

These are about *deployment*, not training. Training cost is mostly unchanged; what they shrink is the KV cache (memory) at inference. This category exploded post-2022 because once GPT-3.5 went into production, KV cache memory became the actual bottleneck for serving cost — not training cost, not training time. The shift from "researchers training things" to "companies serving things at scale" is what made this category important.

### (c) Pareto-frontier improvements — more quality at the same cost

**RoPE. SwiGLU. MoE. Better data.**

These are *not* "efficiency wins" in the conventional sense. They push the Pareto frontier outward rather than moving along it. Calling them efficiency wins undersells what they actually do — they're "more capability per dollar," which is genuinely different from "same capability for fewer dollars."

A 7B model with RoPE + SwiGLU + GQA isn't smaller or faster than a 7B model with learned positions + GELU + MHA. It's *better* at the same inference cost. The "modern" version isn't optimizing for efficiency at all — it's optimizing for quality, holding cost fixed.

## The historical arc

The center of gravity has shifted dramatically over the years. What was "the bottleneck to solve" changed:

| Era | Dominant pressure | Representative changes |
|---|---|---|
| **2017–2019** | Training stability at depth | Pre-LN, GELU, residual connections everywhere |
| **2019–2022** | Training scale (bigger models) | Mixed precision (fp16/bf16), Flash Attention, ZeRO, FSDP |
| **2022–2024** | Inference cost (serving GPT-3.5+) | GQA, MQA, KV cache management, paged attention |
| **2024–present** | Long context & multimodal | Ring attention, MLA, hybrid SSMs, RoPE extensions |
| **Underlying all of it** | Data quality | FineWeb-Edu, DCLM, synthetic data, model-curated filtering |

The "modern is about speed/efficiency" framing fits the 2022–2024 era best. Before that, the pressure was on training scale; after that, it's bending toward capability. **Currently in 2026, the most impactful "modern" change isn't an architectural variant at all — it's better data.** A 7B model trained on DCLM beats a 13B model trained on raw CommonCrawl. That's a 2× quality gain with zero architectural change.

## Why quality looks like a constraint, not an objective

A useful frame:

> Labs compete on benchmarks → quality has a *floor* (you can't go below SOTA for your size class). Within that floor, the actual optimization target is **whichever resource is currently scarce** — usually training compute, KV cache memory, or context length. Architectural changes that fail to maintain quality just don't ship.

So when you read "modern model X uses GQA instead of MHA," the implicit story is:
1. Lab realized inference memory was their cost bottleneck.
2. Found a variant that gives 8× KV cache reduction for negligible quality loss.
3. Shipped it.

You don't see the variants that *did* sacrifice quality too much — they got cut. **Survivorship bias in architecture papers is real.** For every accepted "modern" technique, there are 5-10 that got tried, failed quality, and never made it into a production model.

This is why every architectural "improvement" you see looks like "free lunch" — only the free lunches survive. There were plenty of un-free lunches that were quietly abandoned.

## The interesting counter-trend: "make it bigger"

One trend that genuinely goes against efficiency: **MoE (Mixture of Experts).**

Models like Mixtral, DeepSeek-V3, GPT-4 (rumored) have *more* total parameters (DeepSeek-V3: 671B!) but only activate a small subset per token (~37B active). This is the opposite of efficiency in *total* compute — it's deliberately spending more on training to get a model that's cheaper *per inference token* at high quality.

So **"spend more training compute, get smarter model that costs same to serve"** is also a force. It just looks like "more efficient at inference" if you only count active params. The total parameter count (and total training cost) is going up, not down.

This is a meaningful pattern: the field is willing to spend arbitrarily more on *training* to save on *inference*, because trained models get served billions of times. Amortize once, save forever.

## The three-category mental model

When you read about a new "modern" architecture choice, ask:

1. **Is this a pure win?** (No quality cost, real efficiency gain.) Just adopt it.
2. **Is this an inference-cost trade?** (Small quality loss, big serving savings.) Depends on whether you care about deployment.
3. **Is this a Pareto improvement?** (More quality at the same cost.) Adopt it if you can afford the implementation effort.

This is more useful than the binary "efficiency vs quality" framing because it tells you whether the change is something you'd benefit from at *your* scale (a hobby project doesn't care about KV cache; a production service does).

## TL;DR

Three takeaways:

1. **"Efficiency without sacrificing quality"** describes one important category of modern change (category 2 above) but undersells categories 1 and 3. Many "modern" choices are quality improvements, not efficiency tradeoffs.

2. **The dominant pressure has shifted over time** — from training stability (2017-2019) → training scale (2019-2022) → inference cost (2022-2024) → long context + capability (2024+). What's "modern" depends on when.

3. **Data has become the biggest lever**, more than any architectural change. FineWeb-Edu, DCLM, synthetic data, and model-curated filtering have moved the SOTA more than any change to attention or MLP design in the same period.

For our codebase (faithful GPT-2 reproduction): we're a snapshot of 2019-era architecture. Modernizing it would mostly look like picking up the *category 1* free wins (Flash Attention ✓ already, RMSNorm, no-bias), maybe one or two *category 2* trades (GQA, but barely matters at 12 heads), and the *category 3* improvements that compose cleanly (RoPE, SwiGLU). Beyond that, the best quality improvement would be on the data side, not the architecture.
