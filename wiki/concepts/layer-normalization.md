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
2. **Exploding gradients**: Weight updates become massive → model diverges; loss $\to \infty$
3. **Internal Covariate Shift**: The input distribution to each layer keeps changing across batches, forcing each layer to constantly re-adapt

## How It Works

At each layer, for a vector $x$:

1. Compute mean $\mu$ and variance $\sigma^2$
2. Normalize:

$$
\hat{x}_i = \frac{x_i - \mu}{\sigma}
$$

Example: $x = [2.5,\ 4.8,\ 1.2,\ 5.5,\ 3.1]$

- $\mu = 3.42,\ \sigma^2 = 2.42$
- $\hat{x} = [-0.59,\ 0.88,\ -1.42,\ 1.33,\ -0.21]$ (mean $\approx 0$, variance $\approx 1$)

## Scale and Shift

After normalization, apply learnable **scale ($\gamma$)** and **shift ($\beta$)**:

$$
\text{output} = \gamma \cdot \hat{x} + \beta = \gamma \cdot \frac{x - \mu}{\sigma + \varepsilon} + \beta
$$

This allows the model to partially **undo** normalization if needed, learning the optimal range per layer. At training start: $\gamma = 1$, $\beta = 0$ (pure normalization).

### Parameter Sharing

| Quantity | Shape | Shared across |
|---|---|---|
| $\mu$, $\sigma^2$ (statistics) | scalar per token | nothing — recomputed per token |
| $\gamma$ (scale) | `(emb_dim,)` | all tokens, all batch examples |
| $\beta$ (shift) | `(emb_dim,)` | all tokens, all batch examples |

$\gamma$ and $\beta$ are **per-feature, not per-token**: every token in the sequence and every example in the batch is transformed by the same $(\gamma, \beta)$ vectors. Within one token's `emb_dim` features, each feature has its own scalar $\gamma_i$, $\beta_i$. GPT-2 124M LayerNorm has only $2 \times 768 = 1536$ params per layer, regardless of `seq_len` or `batch_size`. See [[layernorm-scale-shift-sharing]] for full walkthrough.

### Layer Count in a Pre-LN Stack

Each transformer block contains **two** LayerNorms (one before attention, one before FFN), plus **one final** LayerNorm before the output head:

$$
\text{total\_layernorms} = 2 \times n_{\text{layers}} + 1
$$

For GPT-2 124M: $2 \times 12 + 1 = 25$ LayerNorms $\to 25 \times 1536 = 38{,}400$ params (~0.03% of 124M). Each block has its own independent $(\gamma, \beta)$ — they are not shared across depth, because each block sits at a different point in the network and sees a different activation distribution. See [[layernorm-count-gpt2]] for per-model-size table and verification code.

## Pre-LN vs Post-LN

| | Order | Properties |
|---|---|---|
| **Post-LN** (original paper) | Sublayer → Residual Add → Normalize | Harder to train; needs LR warmup |
| **Pre-LN** (modern standard) | Normalize → Sublayer → Residual Add | Easier to train; more stable |

All modern models (GPT-3, LLaMA, BART, T5) use **Pre-LN**.

## Residual Connections

After layer norm + sublayer output, the **original input $X$ is added back** (skip connection):

$$
\text{output} = X + \text{sublayer}(\text{LayerNorm}(X))
$$

This ensures gradients can flow through very deep networks and the model doesn't forget the original representation.

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[residual-connections]]
