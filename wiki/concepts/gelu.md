---
title: GELU Activation
type: concept
tags: [activation, gelu, feed-forward, pytorch]
sources: 1
updated: 2026-04-14
---

## GELU Activation

**Summary**: Gaussian Error Linear Unit — a smooth, probabilistically-motivated activation function used in GPT-2's FFN layers instead of ReLU.

---

## Formula

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where Φ(x) is the cumulative distribution function of the standard normal distribution.

In practice, a fast approximation is used:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right]\right)$$

PyTorch exposes both: `torch.nn.functional.gelu(x)` (exact) and the approximation via `approximate='tanh'`.

---

## Why GELU instead of ReLU?

| Property              | ReLU                     | GELU                            |
| --------------------- | ------------------------ | ------------------------------- |
| Shape                 | Hard threshold at 0      | Smooth, soft gate               |
| Gradient at 0         | Undefined / dead neurons | Non-zero, always differentiable |
| Negative inputs       | Always 0                 | Small negative output allowed   |
| Empirical performance | Strong baseline          | Better on language/vision tasks |

GELU "softly gates" inputs — values far below 0 are suppressed, values far above 0 pass through, but near 0 the transition is smooth. This smoothness helps gradient flow in deep networks.

---

## Where it appears in transformers

Inside the [[feed-forward-network|FFN]] sub-block of each transformer layer:

```
768 → Linear(768, 3072) → GELU → Linear(3072, 768)
```

The expansion (×4) followed by GELU is the standard GPT-2 pattern. See [[gpt2-from-scratch]] for the exact implementation.

---

## Related

- [[feed-forward-network|Feed-Forward Network (FFN)]]
- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[pytorch-nn-building-blocks|PyTorch nn Building Blocks]]
- [[layer-normalization]]
