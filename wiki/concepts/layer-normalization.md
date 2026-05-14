---
title: Layer Normalization
type: concept
tags: [normalization, training-stability, pre-ln, post-ln, gradient]
sources: 1
updated: 2026-04-13
verified_against: Decoder_archtecture, 2026-04-13
confidence: high
---

## Layer Normalization

**Summary**: A technique that normalizes the activations at each layer to stabilize training, solving vanishing/exploding gradients and internal covariate shift in deep networks.

## Problems It Solves

Deep neural networks suffer from three training problems:

1. **Vanishing gradients**: Early layers receive near-zero gradients → almost no learning
2. **Exploding gradients**: Weight updates become massive → model diverges; loss → ∞
3. **Internal Covariate Shift**: The input distribution to each layer keeps changing across batches, forcing each layer to constantly re-adapt

## How It Works

At each layer, for a vector x:
1. Compute mean μ and variance σ²
2. Normalize: x̂ᵢ = (xᵢ − μ) / σ

Example: x = [2.5, 4.8, 1.2, 5.5, 3.1]
- μ = 3.42, σ² = 2.42
- x̂ = [−0.59, 0.88, −1.42, 1.33, −0.21]  (mean ≈ 0, variance ≈ 1)

## Scale and Shift

After normalization, apply learnable **scale (γ)** and **shift (β)**:

`output = γ · x̂ + β = γ · (x − μ)/(σ + ε) + β`

This allows the model to partially **undo** normalization if needed, learning the optimal range per layer. At training start: γ=1, β=0 (pure normalization).

### Parameter Sharing

| | Shape | Shared across |
|---|---|---|
| `μ`, `σ²` (statistics) | scalar per token | nothing — recomputed per token |
| `γ` (scale) | `(emb_dim,)` | all tokens, all batch examples |
| `β` (shift) | `(emb_dim,)` | all tokens, all batch examples |

`γ` and `β` are **per-feature, not per-token**: every token in the sequence and every example in the batch is transformed by the same `(γ, β)` vectors. Within one token's `emb_dim` features, each feature has its own scalar `γᵢ`, `βᵢ`. GPT-2 124M LayerNorm has only `2 × 768 = 1536` params per layer, regardless of `seq_len` or `batch_size`. See [[layernorm-scale-shift-sharing]] for full walkthrough.

### Layer Count in a Pre-LN Stack

Each transformer block contains **two** LayerNorms (one before attention, one before FFN), plus **one final** LayerNorm before the output head:

```
total_layernorms = 2 × n_layers + 1
```

For GPT-2 124M: `2 × 12 + 1 = 25` LayerNorms → `25 × 1536 = 38,400` params (~0.03% of 124M). Each block has its own independent `(γ, β)` — they are not shared across depth, because each block sits at a different point in the network and sees a different activation distribution. See [[layernorm-count-gpt2]] for per-model-size table and verification code.

## Pre-LN vs Post-LN

| | Order | Properties |
|---|---|---|
| **Post-LN** (original paper) | Sublayer → Residual Add → Normalize | Harder to train; needs LR warmup |
| **Pre-LN** (modern standard) | Normalize → Sublayer → Residual Add | Easier to train; more stable |

All modern models (GPT-3, LLaMA, BART, T5) use **Pre-LN**.

## Residual Connections

After layer norm + sublayer output, the **original input X is added back** (skip connection):
`output = X + sublayer(LayerNorm(X))`

This ensures gradients can flow through very deep networks and the model doesn't forget the original representation.

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[residual-connections]]
