---
title: Why .bool() on the Causal Mask
type: query
tags: [causal-masking, pytorch, register_buffer, masked_fill, gpt2]
updated: 2026-05-14
---

## Why `.bool()` on the Causal Mask

**Summary**: `.bool()` is required by `masked_fill` which only accepts boolean tensors, and it saves memory compared to integer tensors.

---

## What Is the Causal Mask?

In a language model, each token should only be able to attend to itself and the tokens that came before it — not the tokens that come after. This is called **causal masking** (also called an autoregressive mask).

The causal mask is a matrix of True/False values that tells the attention mechanism which positions to block:
- `True` = this is a future token → block it (replace with -infinity)
- `False` = this is a past or current token → allow it

---

## How It Is Created

```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(cfg['context_length'], cfg['context_length']),
               diagonal=1).bool())
```

**Step 1 — `torch.ones(...)`:** Creates a matrix of all 1s with shape `(context_length, context_length)`.

**Step 2 — `torch.triu(..., diagonal=1)`:** Keeps only the upper triangle (above the main diagonal), setting everything else to 0:

```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

The 1s represent future positions that should be blocked.

**Step 3 — `.bool()`:** Converts 0s and 1s to False and True:

```
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
```

---

## Why `.bool()` Is Required

The mask is used with `masked_fill`:

```python
att_scores.masked_fill(self.causal_mask[:num_tokens, :num_tokens], -torch.inf)
```

`masked_fill` fills every position where the mask is `True` with `-infinity`. It strictly requires a **boolean tensor** as input. If you pass an integer tensor (0s and 1s), PyTorch raises a `RuntimeError`.

So `.bool()` is not optional — it is required for the code to run.

---

## Why `.bool()` Also Saves Memory

| Dtype | Bytes per element | 1024×1024 mask |
|---|---|---|
| float32 | 4 bytes | 4 MB |
| int32 | 4 bytes | 4 MB |
| bool | 1 byte | **1 MB** |

The mask lives in GPU memory for the entire training run. Using bool instead of float or int gives a 4× memory saving — which matters when context lengths are large.

---

## The Full Flow During a Forward Pass

```
# Setup (once, at model creation):
torch.triu(ones, diagonal=1)     → integer tensor (1 = future, 0 = past)
.bool()                          → bool tensor   (True = future, False = past)
register_buffer(...)             → non-trainable, moves to GPU with the model

# At every forward pass:
att_scores = Q @ Kᵀ / √d_k      → raw attention scores
masked_fill(mask[:T,:T], -inf)   → future positions set to -infinity
softmax(att_scores)              → -infinity becomes 0.0 (zero attention weight)
```

The result: each token can only attend to itself and previous tokens. Future tokens are invisible.

---

## Why `mask[:num_tokens, :num_tokens]`?

The mask is pre-built for the full `context_length` (e.g., 256×256). But the actual input sequence may be shorter. Slicing `[:num_tokens, :num_tokens]` trims the mask to match the actual sequence length — no wasted computation, no index errors.

---

## Related

- [[causal-masking]]
- [[multi-head-attention]]
- [[gpt2-from-scratch]]
- [[pytorch-nn-building-blocks]]
