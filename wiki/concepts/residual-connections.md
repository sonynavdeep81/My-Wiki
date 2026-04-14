---
title: Residual Connections
type: concept
tags: [residual, skip-connections, gradient, training-stability, transformer]
sources: 1
updated: 2026-04-14
---

## Residual Connections

**Summary**: Skip connections that add a layer's input directly to its output, preventing vanishing gradients and allowing information to bypass sublayers in deep networks.

## What They Are

Also called **shortcut connections** or **skip connections**. After a sublayer (attention or FFN), the original input X is added back to the sublayer output:

```
output = X + sublayer(X)          # Post-LN style
output = X + sublayer(LayerNorm(X))  # Pre-LN style (modern)
```

In PyTorch (from the GPT-2 implementation):
```python
# Attention block
shortcut = x
x = self.ln1(x)
x = self.att(x)
x = self.dropout(x)
x = x + shortcut    # ← residual add

# FFN block
shortcut = x
x = self.ln2(x)
x = self.ff(x)
x = self.dropout(x)
x = x + shortcut    # ← residual add
```

## Why They Matter

1. **Prevent vanishing gradients**: During backpropagation, gradients flow directly through the skip path (+1 gradient highway), bypassing the sublayer. Even if the sublayer's gradient is near zero, learning continues.

2. **Preserve original information**: The model doesn't have to re-learn the identity mapping — if a layer learns nothing useful, the residual ensures the representation isn't destroyed.

3. **Enable very deep networks**: Without residuals, stacking 12+ transformer blocks would be unstable. With them, GPT-3 trains 96 layers, and modern models train hundreds.

## Interaction with Layer Normalization

Residual connections and [[layer-normalization]] are always paired:

- **Post-LN** (original Transformer): sublayer → residual add → normalize
- **Pre-LN** (modern standard): normalize → sublayer → residual add

Pre-LN is more stable because the residual stream is normalized before entering each sublayer, keeping gradients well-behaved from the start of training.

## Related

- [[layer-normalization]]
- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[feed-forward-network]]
