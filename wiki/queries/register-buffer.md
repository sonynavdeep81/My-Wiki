---
title: Why Use register_buffer?
type: query
tags: [pytorch, register_buffer, nn.Module, causal-masking, gpt2]
updated: 2026-05-14
---

## Why Use register_buffer?

**Summary**: `register_buffer` registers a fixed tensor with `nn.Module` so it automatically moves to the correct device with the model, is saved and restored in checkpoints, and is never touched by the optimizer.

---

## The Problem It Solves

Inside `MultiHeadAttention.__init__`, we create the causal mask:

```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(cfg['context_length'], cfg['context_length']),
               diagonal=1).bool())
```

Why not just assign it as a plain attribute?

```python
self.causal_mask = torch.triu(...).bool()   # plain tensor — this causes problems
```

If you use a plain attribute, three things go wrong:

---

## Problem 1 — Device Mismatch

When you move the model to GPU:
```python
model = GPT2Model(cfg).to('cuda')
```

All `nn.Parameter` tensors move to GPU automatically. But a plain `self.causal_mask` stays on CPU. Then in the forward pass:

```python
att_scores.masked_fill(self.causal_mask[:T, :T], -torch.inf)
# att_scores is on GPU, causal_mask is on CPU → RuntimeError!
```

With `register_buffer`, the mask moves to GPU automatically when you call `.to('cuda')`.

---

## Problem 2 — Lost in Checkpoints

When you save the model:
```python
torch.save(model.state_dict(), 'model.pt')
```

`state_dict()` includes all `nn.Parameter` tensors AND all registered buffers. A plain attribute is invisible to `state_dict()` — it is not saved.

When you reload:
```python
model.load_state_dict(torch.load('model.pt'))
```

The mask is missing. You would need to manually recreate it every time you load a checkpoint. `register_buffer` saves and restores it automatically.

---

## Problem 3 — Optimizer Interference

If you accidentally used `nn.Parameter` for the mask:
```python
self.causal_mask = nn.Parameter(torch.triu(...).bool())
```

The optimizer would try to update the mask values during training — which is nonsensical. A causal mask should be fixed for the entire training run. `register_buffer` registers the tensor as non-trainable, so the optimizer never touches it.

---

## The Three-Way Classification

Every tensor in a PyTorch model falls into one of three categories:

```
nn.Parameter     → learnable, in parameters(), updated by optimizer
                   Examples: W_Q, W_K, W_V, LayerNorm scale/shift, embeddings

register_buffer  → fixed, in state_dict(), moves with model, NOT in parameters()
                   Examples: causal_mask, running statistics in BatchNorm

plain attribute  → completely invisible to PyTorch
                   Examples: self.n_heads = 12  (just a Python integer)
```

`register_buffer` is the middle ground — PyTorch is aware of it (moves it, saves it), but the optimizer ignores it.

---

## Behavior Summary

| Action | Plain attribute | register_buffer | nn.Parameter |
|---|---|---|---|
| `model.to('cuda')` | Stays on CPU | Moves to GPU ✓ | Moves to GPU ✓ |
| `model.state_dict()` | Not included | Included ✓ | Included ✓ |
| `model.parameters()` | Not included | Not included | Included ✓ |
| Optimizer updates it | No | No | Yes |

---

## In Practice

```python
# At model creation:
model = GPT2Model(cfg).to('cuda')
# self.causal_mask is now on cuda automatically — no manual .to('cuda') needed

# Forward pass — no device mismatch:
att_scores.masked_fill(self.causal_mask[:T, :T], -torch.inf)   # both on cuda ✓

# Saving and loading — mask preserved automatically:
torch.save(model.state_dict(), 'model.pt')
model2 = GPT2Model(cfg)
model2.load_state_dict(torch.load('model.pt'))
# model2.causal_mask is correctly restored ✓
```

---

## Related

- [[causal-masking]]
- [[causal-mask-bool]]
- [[gpt2-from-scratch]]
- [[model-parameters-numel]]
