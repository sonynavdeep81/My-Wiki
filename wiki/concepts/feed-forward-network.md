---
title: Feed-Forward Network (FFN)
type: concept
tags: [ffn, transformer, gelu, non-linearity, neural-network]
sources: 1
updated: 2026-04-13
---

## Feed-Forward Network (FFN)

**Summary**: A two-layer MLP with GELU activation applied independently to each token after multi-head attention, adding non-linear feature transformation to the transformer block.

## Structure

```
Input: (T, 768)
  ↓ Linear Layer 1: W₁(768×3072) + b₁   [expansion: 4×]
Output: (T, 3072)
  ↓ GELU Activation
Output: (T, 3072)
  ↓ Linear Layer 2: W₂(3072×768) + b₂   [projection back]
Output F: (T, 768)
```

- Expansion factor = **4×** (768 → 3072 for GPT-2)
- **GELU** (Gaussian Error Linear Unit) is the non-linearity; smoother than ReLU
- Output shape matches input — passes straight through to the next residual connection

## Role in the Transformer

- All **token mixing** already happened in [[multi-head-attention]] (attention weighted sum of value vectors)
- The FFN processes **each token's vector independently** — no cross-token communication here
- It adds deeper, non-linear features on top of the context-aware representation from attention
- Analogy: attention = "which tokens are relevant?"; FFN = "what does this mean, deeper?"

## Per-Block, Per-Layer

Each of the Nx=12 transformer blocks has its own FFN with independent learned weights. This is where a large portion of the model's "knowledge" is believed to be stored.

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[layer-normalization]]
