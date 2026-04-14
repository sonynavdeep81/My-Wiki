---
title: Weight Tying
type: concept
tags: [weight-tying, embeddings, output-head, parameters, gpt2]
sources: 1
updated: 2026-04-13
---

## Weight Tying

**Summary**: A technique where the token embedding matrix and the output projection head share the same weight tensor, reducing parameters and improving training stability.

## What It Is

In a decoder-only LLM:
- **Input side**: `tok_emb` maps token IDs → 768-dim vectors (shape: 50,257 × 768)
- **Output side**: `out_head` maps 768-dim vectors → logits over vocabulary (shape: 768 × 50,257)

These two matrices are transposes of each other conceptually. Weight tying makes them literally the **same tensor**:

```python
# Both point to the same underlying data
assign(model.tok_emb.weight, params['wte'])
assign(model.out_head.weight, params['wte'])
```

## Why It Works

The embedding matrix learns: "token 42 lives at position [0.3, -0.1, ...]"  
The output head learns: "position [0.3, -0.1, ...] means token 42 is likely"

These are inverse operations — it's natural for them to share weights. The model learns both simultaneously from a single set of parameters.

## Benefits

1. **Parameter reduction**: saves 50,257 × 768 ≈ 38.6M parameters
2. **Training signal**: each gradient update on the output head also improves the input embeddings, and vice versa — faster convergence
3. **Symmetry**: tokens that are semantically similar end up close in embedding space, which also makes them easy to predict as outputs

## Implementation Note

`out_head` in GPT-2 uses `bias=False`:
```python
self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
```
Modern LLMs universally drop this bias — it's a redundant parameter at this layer.

## Related

- [[embeddings]]
- [[gpt2-from-scratch]]
- [[decoder-only-architecture]]
