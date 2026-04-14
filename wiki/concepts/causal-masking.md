---
title: Causal Masking
type: concept
tags: [causal-mask, look-ahead-mask, attention, decoder, autoregressive]
sources: 1
updated: 2026-04-14
---

## Causal Masking

**Summary**: A mechanism that prevents tokens from attending to future positions during training, enforcing autoregressive structure so each token can only see itself and previous tokens.

## The Problem

During training, all tokens are processed in parallel (for efficiency). But for next-token prediction to work correctly, token at position i must **not** see tokens at positions i+1, i+2, ... — otherwise the model would just copy the answer.

## The Solution: Causal (Look-Ahead) Mask

A triangular binary mask is applied to the attention score matrix **before** softmax:

```
Upper triangle → -∞    (future positions: blocked)
Lower triangle + diagonal → 0  (past + current: allowed)
```

Example for sequence length 4:

```
         Every  effort  takes   you
Every  [  0     -∞      -∞     -∞  ]
effort [  0      0      -∞     -∞  ]
takes  [  0      0       0     -∞  ]
you    [  0      0       0      0  ]
```

After adding this to the scaled attention scores, softmax converts -∞ → 0, so future tokens get zero attention weight.

## Implementation

In PyTorch, registered as a non-trainable buffer (moves to GPU automatically):

```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(context_length, context_length),
               diagonal=1).bool())

# In forward():
scores.masked_fill_(self.causal_mask[:T, :T], float('-inf'))
```

`torch.triu(..., diagonal=1)` creates the upper triangle excluding the diagonal.

## Why register_buffer?

- Not a trainable parameter — the mask doesn't learn
- Must move to the same device as the model (CPU/GPU)
- `register_buffer` handles both: excluded from `parameters()`, included in `state_dict()` and device moves

## Training vs Inference

| | Training | Inference |
|---|---|---|
| Tokens processed | All in parallel | One new token per step |
| Mask needed? | Yes — prevents future look-ahead | Yes — same mask applies |
| Why it matters | Enables teacher forcing with correct gradients | Maintains autoregressive generation |

The mask is what allows training to be parallel while the model still learns to generate sequentially.

## Related

- [[multi-head-attention]]
- [[decoder-only-architecture]]
- [[transformer-architecture]]
- [[residual-connections]]
