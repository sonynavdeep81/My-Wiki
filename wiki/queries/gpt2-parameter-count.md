---
title: GPT-2 Parameter Count — 124M vs 162M
type: query
tags: [gpt2, parameters, weight-tying, pytorch, numel]
sources: 1
updated: 2026-04-23
sources: 1 (raw/gpt2_decoder.py — verified assign() uses copy_(), no tying line in __init__)
---

## GPT-2 Parameter Count — 124M vs 162M

**Summary**: GPT-2 small has ~124M unique parameters; `model.parameters()` reports ~162M because weight-tied tok_emb/out_head are double-counted.

## Component Breakdown

| Component | Calculation | Parameters |
|---|---|---|
| Token embeddings (tok_emb) | 50,257 × 768 | ≈ 38.6M |
| Positional embeddings | 256 × 768 | ≈ 0.2M |
| 12 × Attention (Q,K,V,W_O) | 12 × 4 × 768² | ≈ 28.3M |
| 12 × FFN (768→3072→768) | 12 × 2 × 768×3072 | ≈ 56.6M |
| LayerNorms | negligible | — |
| **Total (unique)** | | **≈ 124M** |

## Why This Implementation Always Shows 162M

In the from-scratch `__init__`, `out_head` is created as a **separate** tensor — no weight tying line exists:

```python
self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)  # own tensor
# missing: self.out_head.weight = self.tok_emb.weight
```

When loading OpenAI weights, `assign()` uses `copy_()` — copies values, does NOT make them the same tensor:

```python
assign(model.tok_emb.weight, params['wte'])   # copies values into tok_emb
assign(model.out_head.weight, params['wte'])  # copies values into out_head — still separate tensor
```

```python
model.out_head.weight is model.tok_emb.weight  # False — different tensors, same values
```

Result: `out_head` is always a separate 38.6M tensor → `model.parameters()` counts it independently → **162M always**.

## True Weight Tying (not in this implementation)

```python
# In __init__, after creating both:
self.out_head.weight = self.tok_emb.weight  # same tensor object

# Verify:
model.out_head.weight is model.tok_emb.weight  # True → 124M unique params
```

## Getting the Right Count

```python
# Counts separately → 162M (this implementation)
sum(p.numel() for p in model.parameters())

# Deduplicated by tensor identity → 124M (only if truly tied)
sum(p.numel() for p in set(model.parameters()))
```

## Related

- [[weight-tying]]
- [[gpt2-from-scratch]]
- [[model-parameters-numel]]
- [[embeddings]]
