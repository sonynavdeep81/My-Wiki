---
title: Dropout
type: concept
tags: [regularization, dropout, pytorch, training]
sources: 2
updated: 2026-05-14
verified_against: gpt2_decoder, 2026-05-14
confidence: high
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
  ├── dropout: nn.Dropout(0.1)              ← (1) after embedding sum
  └── trf_blocks: 12 × TransformerBlock
        ├── att: MultiHeadAttention
        │     └── dropout(0.1) on att_weights  ← (2) on attention weights (post-softmax, pre-V multiply)
        ├── dropout(0.1) on att output         ← (3) after MHA, before residual add
        └── dropout(0.1) on ffn output         ← (4) after FFN, before residual add
```

**Per transformer block: 3 dropout sites.** Plus 1 after the embedding sum.

| Site | Where | What it zeros |
|---|---|---|
| (1) post-embedding | `GPT2Model.forward` | tok_emb + pos_emb sum |
| (2) on attention weights | inside `MultiHeadAttention` after softmax | randomly drops token-to-token connections |
| (3) post-attention | `TransformerBlock.forward` after `att(x)` | attention sublayer's output before residual |
| (4) post-FFN | `TransformerBlock.forward` after `ff(x)` | FFN sublayer's output before residual |

GPT-2 124M total dropout sites: `1 + 12 × 3 = 37`. All share the same `drop_rate=0.1`. All disabled automatically at inference via `model.eval()`.

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
