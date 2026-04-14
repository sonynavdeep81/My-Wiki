---
title: Multi-Head Attention
type: concept
tags: [attention, transformer, self-attention, causal-masking, qkv]
sources: 1
updated: 2026-04-13
---

## Multi-Head Attention

**Summary**: The core mechanism of the transformer that allows each token to attend to all previous tokens simultaneously, using multiple parallel "heads" to capture different types of relationships.

## Q, K, V Matrices

Given input X of shape (T, d_model) — e.g., (4, 768):

```
Q = X · W_Q    {(4,768) · (768,768) → (4,768)}
K = X · W_K    {(4,768) · (768,768) → (4,768)}
V = X · W_V    {(4,768) · (768,768) → (4,768)}
```

W_Q, W_K, W_V are **learned weight matrices**.

## Multiple Heads

For GPT-2 with d_model=768 and 12 heads: each head gets d_k = 768/12 = **64 dimensions**.

Q, K, V are split into 12 heads:
- Shape before split: (4, 768)
- Shape after split: (4, 12, 64)
- Reshaped for parallel computation: **(12, 4, 64)**

## Attention Scores

For each head: `Scores = Q · K^T`  
`{(12,4,64) · (12,64,4) → (12,4,4)}`

Each score (i,j) represents how much token i should attend to token j — the dot product of their query and key vectors. Higher = more related.

## Scaling

Scores are divided by √d_k = √64 = 8 to prevent softmax saturation with large values.

## Causal Mask (Decoder)

Future positions are set to −∞ before softmax:

```
Causal mask: upper triangle = −∞, lower triangle (including diagonal) = 0
```

After adding the mask, softmax converts −∞ to 0 — each token can only attend to itself and previous tokens.

## Attention Weights

`A = softmax(masked_scores)`  shape: (12, 4, 4)  
Each row sums to 1.0 — a probability distribution over visible tokens.

## Dropout

Applied to the attention weight matrix during training (p=0.1 → 10% zeroed). Remaining values scaled by 1/(1−p) = 1.1111 to keep the sum stable. Dropout is **off during inference**.

## Contextual Vectors

`C = A · V`  `{(12,4,4) · (12,4,64) → (12,4,64)}`

Reshape: (12,4,64) → (4,12,64) → **(4, 768)** (concatenate all heads)

Then projected: `output = C · W_O`  `{(4,768) · (768,768) → (4,768)}`

W_O mixes information across all 12 heads into a single unified representation.

## Why Multiple Heads?

- Each head looks at the sequence from a **different angle** simultaneously
- Concatenation collects all 12 perspectives into one place
- W_O merges them into a single unified understanding

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[kv-caching]]
- [[causal-masking]]
- [[attention-is-all-you-need]]
