---
title: Why Use register_buffer?
type: query
tags: [pytorch, register_buffer, nn.Module, causal-masking, gpt2]
updated: 2026-04-14
---

## Why Use `register_buffer`?

**Summary**: `register_buffer` registers a fixed tensor with `nn.Module` so it automatically moves with the model to any device, is saved/restored in checkpoints, and is never touched by the optimizer.

---

## The Problem Without It

If you assign the causal mask as a plain attribute:

```python
self.causal_mask = torch.triu(...).bool()  # plain tensor
```

Three things break:

| Problem | What happens |
|---|---|
| **Device mismatch** | Mask stays on CPU when model moves to GPU → `RuntimeError` at `masked_fill_` |
| **Checkpoint missing** | `state_dict()` excludes it → mask is lost on save/load |
| **Optimizer interference** | If `nn.Parameter` were used instead, optimizer would try to train it — nonsensical for a fixed mask |

---

## What `register_buffer` Does

```python
self.register_buffer('causal_mask', torch.triu(...).bool())
```

| Behaviour | Detail |
|---|---|
| `model.to('cuda')` | Mask moves to GPU automatically with all weights |
| `model.state_dict()` | Mask included → saved and restored with checkpoints |
| `model.parameters()` | Mask excluded → optimizer never updates it |
| `model.eval()` | No effect needed — not a learned value |

---

## The Three-Way Classification

Every tensor in your GPT-2 implementation falls into one of three categories:

```
nn.Parameter      → learnable, in parameters(), updated by optimizer
                    e.g. W_Q, W_K, W_V, LayerNorm γ/β, embeddings

register_buffer   → fixed, in state_dict(), moves with model
                    e.g. causal_mask

plain attribute   → invisible to PyTorch entirely
  self.x = ...      e.g. self.n_heads = 12 (just a Python int)
```

`register_buffer` is the middle ground — PyTorch-aware but not trainable.

---

## In Practice

```python
# Mask follows model to whatever device automatically
model = GPT2Model(cfg).to('cuda')
# self.causal_mask is now on cuda — no manual .to('cuda') needed

# Forward pass — no device mismatch
scores.masked_fill_(self.causal_mask[:T, :T], float('-inf'))

# Save/load — mask is preserved in checkpoint
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))  # mask restored correctly
```

---

## Related

- [[causal-masking]]
- [[causal-mask-bool|Why .bool() on the Causal Mask]]
- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[pytorch-nn-building-blocks|PyTorch nn Building Blocks]]
