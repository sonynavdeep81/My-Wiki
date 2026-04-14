---
title: Layer Normalization
type: concept
tags: [normalization, training-stability, pre-ln, post-ln, gradient]
sources: 1
updated: 2026-04-13
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
