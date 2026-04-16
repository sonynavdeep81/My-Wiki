---
title: Feed-Forward Network (FFN)
type: concept
tags: [ffn, transformer, gelu, non-linearity, neural-network]
sources: 2
updated: 2026-04-13
---

## Feed-Forward Network (FFN)

**Summary**: A two-layer MLP applied independently to each token after multi-head attention, adding non-linear feature transformation to the transformer block. Activation varies by implementation: ReLU in the original Transformer paper, GELU in GPT-2.

## Structure

```
Input: (T, d_model)
  ↓ Linear Layer 1: W₁ + b₁   [expansion: 4×]
  ↓ Activation
  ↓ Linear Layer 2: W₂ + b₂   [projection back]
Output: (T, d_model)
```

Output shape matches input — passes straight through to the next residual connection.

### In your GPT-2 implementation
- d_model=768, d_ff=3072 (4×)
- Activation: **[[gelu|GELU]]**
- 12 FFN blocks (one per decoder layer)

### In the Attention Is All You Need paper
- d_model=512, d_ff=2048 (4×)
- Activation: **ReLU** — `FFN(x) = max(0, xW₁+b₁)W₂+b₂`
- 6 FFN blocks each in encoder and decoder

## Role in the Transformer

- All **token mixing** already happened in [[multi-head-attention]] (attention weighted sum of value vectors)
- The FFN processes **each token's vector independently** — no cross-token communication here
- It adds deeper, non-linear features on top of the context-aware representation from attention
- Analogy: attention = "which tokens are relevant?"; FFN = "what does this mean, deeper?"

## Per-Block, Per-Layer

Each transformer block has its own FFN with independent learned weights. This is where a large portion of the model's "knowledge" is believed to be stored.

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[layer-normalization]]
- [[gelu|GELU Activation]]
- [[dropout]]
