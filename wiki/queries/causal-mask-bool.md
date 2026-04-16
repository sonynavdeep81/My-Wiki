---
title: Why .bool() on the Causal Mask
type: query
tags: [causal-masking, pytorch, register_buffer, masked_fill, gpt2]
updated: 2026-04-14
---

## Why `.bool()` on the Causal Mask

**Summary**: `.bool()` is required by `masked_fill_` which only accepts boolean tensors, and it halves memory usage compared to integer tensors.

```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(cfg['context_length'], cfg['context_length']),
               diagonal=1).bool())
```

---

## What `.bool()` Does to the Tensor

`torch.triu(..., diagonal=1)` produces integer 0s and 1s:

```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

After `.bool()`:

```
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
```

`True` = future position (blocked), `False` = past/current position (allowed).

---

## Purpose 1 — `masked_fill_` requires a BoolTensor

The mask is never added to scores as a value. It is used to **select positions to overwrite**:

```python
scores.masked_fill_(self.causal_mask[:T, :T], float('-inf'))
```

`masked_fill_` fills every position where the mask is `True` with the given value (`-inf`). It strictly requires a boolean tensor — passing an integer tensor raises a `RuntimeError`.

---

## Purpose 2 — Memory Efficiency

| Dtype | Bytes per element | 1024×1024 mask |
|---|---|---|
| float32 | 4 bytes | 4 MB |
| int32 | 4 bytes | 4 MB |
| bool | 1 byte | **1 MB** |

The mask lives in GPU memory for the entire training run, so the 4× saving matters.

---

## Full Forward-Pass Flow

```
torch.triu(ones, diagonal=1)        → int tensor  (upper triangle = 1)
.bool()                             → bool tensor (upper triangle = True)
register_buffer(...)                → non-trainable, moves to GPU with model

# At each forward pass:
scores = Q @ K.T / sqrt(d_k)
scores.masked_fill_(mask[:T,:T], -inf)
#   True  positions → -inf  (future tokens, blocked)
#   False positions → unchanged (past/current tokens, visible)
softmax(scores)                     → -inf becomes 0.0 (zero attention weight)
```

---

## Related

- [[causal-masking]]
- [[multi-head-attention]]
- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[pytorch-nn-building-blocks|PyTorch nn Building Blocks]]
