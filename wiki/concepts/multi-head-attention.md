---
title: Multi-Head Attention
type: concept
tags: [attention, transformer, self-attention, causal-masking, qkv]
sources: 1
updated: 2026-04-17
---

## Multi-Head Attention

**Summary**: Each token attends to all prior tokens via learned Q/K/V projections split across h parallel heads; outputs concatenated and projected via W_O.

> Numbers: GPT-2 impl (d_model=768, h=12) unless labelled. Paper ([[attention-is-all-you-need]]): d_model=512, h=8. Both give d_k=64.

## Forward Pass (GPT-2, T=4)

```
X: (T, 768)
Q = X·W_Q → (T,768);  K = X·W_K → (T,768);  V = X·W_V → (T,768)
Split into 12 heads → (12, T, 64)

scores = Q·Kᵀ / √64          (12,4,4)   — scale prevents softmax saturation
+ causal mask (upper tri = −∞)
A = softmax(scores)            (12,4,4)   — rows sum to 1; future tokens = 0
A = Dropout(A, p=0.1)          — training only
C = A·V                        (12,4,64)
concat → (4,768) → ·W_O → (4,768)
```

## Config Comparison

| | GPT-2 (yours) | Paper |
|---|---|---|
| d_model | 768 | 512 |
| h | 12 | 8 |
| d_k | 64 | 64 |
| qkv_bias (scratch) | False | True |
| qkv_bias (OAI ckpt) | True | — |

## Key Facts

- W_Q, W_K, W_V, W_O are all learned; W_O bias=True
- Causal mask stored as `register_buffer` (non-trainable, device-aware)
- Dropout on attention weights during training; off at inference (`model.eval()`)
- Each head captures different relationship types; W_O merges all heads

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[cross-attention]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[kv-caching]]
- [[causal-masking]]
- [[attention-is-all-you-need]]
