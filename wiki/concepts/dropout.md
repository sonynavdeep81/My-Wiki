---
title: Dropout
type: concept
tags: [regularization, dropout, pytorch, training]
sources: 1
updated: 2026-04-14
---

## Dropout

**Summary**: A regularization technique that randomly zeros out a fraction of activations during training to prevent co-adaptation of neurons and reduce overfitting.

---

## How it works

During **training**: each activation is independently set to 0 with probability `p` (the dropout rate), and the remaining activations are scaled up by `1/(1-p)` to keep the expected value the same.

During **inference**: dropout is disabled — all activations pass through unchanged.

```python
self.drop = nn.Dropout(p=0.1)

# Training: ~10% of values zeroed, rest scaled by 1/0.9
# Inference (model.eval()): no-op
```

PyTorch handles the train/eval switch automatically via `nn.Module`'s `.train()` / `.eval()` — this is one of the key reasons all layers inherit from [[pytorch-nn-building-blocks|nn.Module]].

---

## Where it appears in GPT-2

From the [[gpt2-from-scratch|GPT-2 class hierarchy]]:

```
GPT2Model
  ├── dropout: nn.Dropout(0.1)        ← after embedding sum
  └── trf_blocks: 12 × TransformerBlock
        ├── att: MultiHeadAttention
        │     └── dropout(0.1)        ← after attention weights
        └── ff:  FeedForward
              └── dropout(0.1)        ← after FFN
```

Three dropout points: post-embedding, post-attention softmax, post-FFN. All use `p=0.1` in GPT-2 small.

---

## Effect on training

- Forces the network to learn **redundant representations** — no single neuron can be relied upon
- Acts as an implicit ensemble: each forward pass uses a different subnetwork
- Most effective at moderate depth; very large models (GPT-3+) often use little or no dropout because they are trained with limited epochs and don't overfit in the classical sense

---

## Related

- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[pytorch-nn-building-blocks|PyTorch nn Building Blocks]]
- [[residual-connections]]
- [[layer-normalization]]
- [[label-smoothing|Label Smoothing]]
