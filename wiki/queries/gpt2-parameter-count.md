---
title: GPT-2 Parameter Count — 124M vs 162M
type: query
tags: [gpt2, parameters, weight-tying, pytorch, numel]
sources: 1
updated: 2026-05-14
---

## GPT-2 Parameter Count — 124M vs 162M

**Summary**: GPT-2 small has ~124M unique parameters. Without weight tying, `model.parameters()` reports ~162M because `tok_emb` and `out_head` are counted as two separate 38.6M matrices.

---

## Where Do All the Parameters Come From?

Let's break down every component of GPT-2 small:

| Component | Calculation | Parameters |
|---|---|---|
| Token embeddings (`tok_emb`) | $50{,}257 \times 768$ | $\approx 38.6\text{M}$ |
| Positional embeddings (`pos_emb`) | $256 \times 768$ | $\approx 0.2\text{M}$ |
| $12 \times$ Attention ($W_Q, W_K, W_V, W_O$) | $12 \times 4 \times 768^2$ | $\approx 28.3\text{M}$ |
| $12 \times$ FeedForward ($768 \to 3072 \to 768$) | $12 \times 2 \times 768 \times 3072$ | $\approx 56.6\text{M}$ |
| LayerNorms (scale + shift per block) | negligible | — |
| **Total (unique parameters)** | | $\mathbf{\approx 124\text{M}}$ |

---

## Why the Code Shows 162M

The standard way to count parameters in PyTorch:

```python
sum(p.numel() for p in model.parameters())
```

This iterates through every registered parameter and sums their sizes. The problem: it counts every **tensor** — even if two tensors hold the same data.

Without weight tying, `out_head` is created as a completely **separate** tensor:

```python
self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
# This creates its own 50,257 × 768 tensor — separate from tok_emb
```

So `model.parameters()` finds:
- `tok_emb.weight` $\to 38.6\text{M}$ parameters
- `out_head.weight` $\to 38.6\text{M}$ parameters (separate tensor)

Result: $124\text{M} + 38.6\text{M}$ extra $= \mathbf{\sim 162\text{M}}$

---

## The Difference Between Copying Values and Tying Weights

When loading OpenAI's pretrained weights, this code runs:

```python
assign(model.tok_emb.weight, params['wte'])   # copies values into tok_emb
assign(model.out_head.weight, params['wte'])  # copies values into out_head
```

The `assign()` function uses `copy_()` internally — it copies the **values** but does NOT make both tensors the same object in memory. They are still two separate tensors that happen to contain the same numbers.

```python
model.out_head.weight is model.tok_emb.weight
# → False  (different objects, same values)
```

Because they are different objects, `model.parameters()` counts both → 162M.

---

## True Weight Tying — How to Fix It

Real weight tying makes both names point to the **same tensor object**:

```python
# In __init__, after creating both layers:
self.out_head.weight = self.tok_emb.weight  # same tensor object, two names
```

Now:
```python
model.out_head.weight is model.tok_emb.weight
# → True  (same object)
```

`model.parameters()` deduplicates by tensor identity → counts it only once → **~124M**.

---

## How to Get the True Count Either Way

```python
# Standard count — may double-count tied weights
sum(p.numel() for p in model.parameters())
# → 162M (without tying) or 124M (with tying)

# Deduplicated count — correct regardless of tying
sum(p.numel() for p in set(model.parameters()))
# → 124M always (deduplicates by tensor identity)
```

Using `set()` removes duplicates based on object identity, giving the true unique parameter count even without explicit weight tying.

---

## Summary

| Scenario | tok_emb and out_head | Parameter count |
|---|---|---|
| No weight tying | Two separate tensors (same values after loading) | ~162M |
| Weight tying (`out_head.weight = tok_emb.weight`) | One shared tensor, two names | ~124M |

The original GPT-2 paper reports 124M parameters — this corresponds to the weight-tied version where the embedding matrix is shared.

---

## Related

- [[weight-tying]]
- [[gpt2-from-scratch]]
- [[model-parameters-numel]]
- [[embeddings]]
