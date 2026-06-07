# MoE Dispatch: Three Iterations

A record of the three implementations of `MoELayer.forward()` in `model_v3.py`,
why each was written, and what went wrong or right.

---

## The problem being solved

Each token must be routed to exactly `top_k=2` of the `n_routed=4` experts.
The router produces per-token scores; we select the top-2, renormalize, and
accumulate the weighted expert outputs. The naive implementation of this turns
out to interact badly with `torch.compile`.

---

## Version 1 — Mask-based loop (original, ~7 days)

```python
def forward(self, x):
    B, T, C = x.shape
    N = B * T
    flat_x = x.view(N, C)

    shared_out = sum(e(x) for e in self.shared_experts)

    scores = F.softmax(self.router(flat_x), dim=-1)       # (N, n_routed)
    topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    routed_out = torch.zeros(N, C, device=x.device, dtype=x.dtype)
    for e_id, expert in enumerate(self.routed_experts):
        # Which tokens selected this expert?
        mask = (topk_idx == e_id).any(dim=-1)              # (N,) bool

        if not mask.any():                                  # ← CPU-GPU sync
            continue

        # Gather the scores for this expert for selected tokens
        tok_scores = topk_scores[(topk_idx == e_id).nonzero(as_tuple=True)[0]]

        # Run expert only on selected tokens
        expert_out = expert(flat_x[mask])                  # ← dynamic shape!
        routed_out[mask] += tok_scores.unsqueeze(-1) * expert_out

    lb_loss = self._load_balance_loss(scores, topk_idx)
    return shared_out + routed_out.view(B, T, C), lb_loss
```

### Why it was slow

Two compounding problems:

**1. `mask.any()` — CPU-GPU sync**

`mask.any()` copies a scalar from GPU to CPU to evaluate the `if` condition.
This stalls the GPU pipeline: every micro-step, the CPU waits for the GPU to
finish computing `mask`, then checks it, then resumes dispatching kernels.
With 4 experts × 12 layers × 16 micro-steps = 768 syncs per optimizer step,
this alone adds seconds of wall-clock latency.

**2. `flat_x[mask]` — dynamic tensor shape**

`flat_x[mask]` produces a tensor whose first dimension equals the number of
tokens that chose expert `e_id`. That count changes every step (different
tokens route differently). `torch.compile` embeds tensor shapes as constants
in compiled CUDA kernels:

```
Step 0:  expert 0 receives 8,234 tokens  → compile kernel for shape (8234, 768)
Step 1:  expert 0 receives 8,191 tokens  → shape mismatch → recompile (0.5s)
Step 2:  expert 0 receives 8,309 tokens  → shape mismatch → recompile (0.5s)
...
```

4 experts × 12 layers × 16 micro-steps = 768 potential recompiles per optimizer step.
At 0.5s each, worst case: hundreds of seconds per step → 7-day training.

---

## Version 2 — Sort-based dispatch (attempted fix, still slow)

The idea: sort all tokens by their top-1 expert assignment. Then each expert
receives a contiguous slice of the sorted batch — no masking, no gather.

```python
def forward(self, x):
    B, T, C = x.shape
    N = B * T
    flat_x = x.view(N, C)

    shared_out = sum(e(x) for e in self.shared_experts)

    scores = F.softmax(self.router(flat_x), dim=-1)
    topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    # Sort tokens by their primary expert choice
    primary_expert = topk_idx[:, 0]                        # (N,)
    sort_order = primary_expert.argsort()                  # (N,)
    sorted_x = flat_x[sort_order]                          # (N, C)

    # Count how many tokens go to each expert
    counts = [(primary_expert == i).sum().item()           # ← .item() graph break!
              for i in range(self.n_routed)]

    routed_out = torch.zeros(N, C, device=x.device, dtype=x.dtype)
    start = 0
    for e_id, expert in enumerate(self.routed_experts):
        end = start + counts[e_id]
        if counts[e_id] > 0:
            expert_out = expert(sorted_x[start:end])       # ← dynamic slice!
            routed_out[sort_order[start:end]] += expert_out
        start = end

    lb_loss = self._load_balance_loss(scores, topk_idx)
    return shared_out + routed_out.view(B, T, C), lb_loss
```

### Why it was still slow

The sort-based approach eliminated the boolean mask but introduced two new problems:

**1. `.item()` — graph break**

`(primary_expert == i).sum().item()` pulls a scalar to CPU to use as a Python
integer (`counts[e_id]`). This is a *graph break*: `torch.compile` cannot trace
through it. The compiled graph is split at every `.item()` call, forcing the
compiler to emit multiple smaller graphs and re-enter Python for each one.

**2. `sorted_x[start:end]` — still a dynamic shape**

`start` and `end` are Python integers that change every step (because `counts`
changes). The slice `sorted_x[start:end]` has a shape whose first dimension
equals `counts[e_id]`, which varies. Same root cause as version 1:
torch.compile sees a new shape every step and recompiles.

The sort trick solves contiguity (one matmul per expert in theory), but it
doesn't solve the dynamic-shape problem that torch.compile actually trips on.

---

## Version 3 — Fixed-shape dispatch (current, fast)

Insight: if all tensor shapes are **constant across steps**, torch.compile
compiles once at step 0 and reuses the kernel for every subsequent step.

The key observation is that we don't need to *select* tokens before running
each expert. Instead:

1. Build a `dispatch_w` matrix of shape `(N, n_routed)` — always the same shape.
   `dispatch_w[n, e]` = routing weight if token `n` selected expert `e`, else 0.
2. Run **every expert on the full batch** `flat_x` of shape `(N, C)` — always the same shape.
3. Zero-weight the non-selected outputs: multiply each expert output by its
   column of `dispatch_w`.

No masking. No sorting. No dynamic shapes. No CPU-GPU syncs.

```python
def forward(self, x):
    B, T, C = x.shape
    N = B * T
    flat_x = x.view(N, C)                                  # (N, C) — fixed shape

    shared_out = sum(e(x) for e in self.shared_experts)

    scores = F.softmax(self.router(flat_x), dim=-1)        # (N, n_routed)
    topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    # Build weight matrix — shape (N, n_routed), always constant.
    # dispatch_w[n, e] = routing weight   if token n chose expert e
    #                  = 0                otherwise
    dispatch_w = torch.zeros(N, self.n_routed, device=x.device, dtype=x.dtype)
    dispatch_w.scatter_(1, topk_idx, topk_scores)

    # Run every expert on the full (N, C) batch.
    # Non-selected outputs are zeroed by the 0-weight in dispatch_w.
    # torch.compile sees shape (N, C) every step → compiled once.
    routed_out = torch.zeros(N, C, device=x.device, dtype=x.dtype)
    for e_id, expert in enumerate(self.routed_experts):
        w = dispatch_w[:, e_id].unsqueeze(-1)              # (N, 1) — fixed shape
        routed_out = routed_out + w * expert(flat_x)       # (N, C) — fixed shape

    lb_loss = self._load_balance_loss(scores, topk_idx)
    return shared_out + routed_out.view(B, T, C), lb_loss
```

### Why it's fast

- **No CPU-GPU sync.** `scatter_` is a pure GPU op. No `.any()`, no `.item()`.
- **No dynamic shapes.** `flat_x` is always `(N, C)`. `dispatch_w[:, e_id]` is
  always `(N, 1)`. `torch.compile` compiles once at step 0, reuses every step.
- **No graph breaks.** The Python loop has `n_routed=4` fixed iterations;
  `torch.compile` unrolls it at trace time.

### The trade-off

Running all 4 experts on the full batch instead of 2 means:

```
Version 1/2 (intent):  2 expert forward passes per token
Version 3 (reality):   4 expert forward passes per token  ← 2× more FLOPs
```

With `n_routed=4`, `top_k=2`: 2 extra expert passes per step. Each expert
forward pass is `(N, C) × (C, intermediate)` + `(N, intermediate) × (intermediate, C)`.
The extra cost is real but far smaller than the recompilation it eliminates.

Expected training time with version 3: ~6–8 hours (vs 7 days for v1/v2).

---

## Side-by-side summary

| | Version 1 (mask) | Version 2 (sort) | Version 3 (dispatch_w) |
|---|---|---|---|
| **Expert input shape** | `(selected_N, C)` — varies | `(selected_N, C)` — varies | `(N, C)` — fixed |
| **CPU-GPU sync** | `mask.any()` | `.item()` for counts | None |
| **Graph break** | No | Yes (`.item()`) | No |
| **torch.compile** | Recompiles every step | Recompiles every step | Compiles once |
| **Extra FLOPs** | None (only selected tokens) | None (only selected tokens) | 2× routed experts |
| **Training time (est.)** | ~7 days | ~7 days | ~6–8 hours |

---

## The general lesson

> When using `torch.compile`, **variable tensor shapes are as expensive as
> no compilation at all.** The compiler embeds shapes as constants. A new
> shape = a new compiled kernel = a recompilation penalty on that step.

The fix is always the same: find the variable dimension, and make it constant.
In MoE dispatch, the variable dimension is the per-expert token count. The
solution is to stop selecting tokens before the matmul — instead, run the
full batch and use a weight matrix to zero out the non-selected outputs.

This is the same principle behind other "padding to fixed length" tricks
in production inference engines: a bit of wasted compute buys a lot of
compilation stability.
