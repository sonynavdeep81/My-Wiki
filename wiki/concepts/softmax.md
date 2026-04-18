---
title: Softmax
type: concept
tags: [activation, probability, attention, decoding, normalization]
sources: 1
updated: 2026-04-18
---

## Softmax

**Summary**: Converts a vector of real-valued scores into a probability distribution that sums to 1.

## Formula

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

- Output ∈ (0, 1) for each element; all outputs sum to 1
- Preserves ordering: larger input → larger output
- Sensitive to *differences* between values, not absolute magnitude

## PyTorch

```python
torch.softmax(x, dim=-1)   # along last dimension (token/vocab axis)
```

## Where Used in a Transformer

| Location | Input | Purpose |
|----------|-------|---------|
| Attention scores | Q @ K.T / √d_k | Convert raw scores → attention weights |
| Output head | logits (vocab_size) | Convert logits → token probabilities |
| Temperature decoding | logits / T | Convert scaled logits → sampling probs |

## Numerical Stability

Raw `exp(x)` overflows for large x. In practice:
```
softmax(x) = softmax(x - max(x))   # subtract max before exp
```
PyTorch handles this automatically.

## Softmax vs Sigmoid

| | Softmax | Sigmoid |
|--|---------|---------|
| Output constraint | sums to 1 (mutually exclusive) | each independently ∈ (0,1) |
| Use case | single-label classification, attention | binary / multi-label |

## Related

- [[multi-head-attention]]
- [[decoding-strategies]]
- [[temperature]]
- [[layer-normalization]]
