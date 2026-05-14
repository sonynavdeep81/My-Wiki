---
title: model.parameters() and p.numel() Explained
type: query
tags: [pytorch, nn.Module, parameters, numel, weight-tying, gpt2]
updated: 2026-05-14
---

## model.parameters() and p.numel() Explained

**Summary**: `model.parameters()` recursively yields every trainable parameter in the model. `p.numel()` counts the scalar elements in one parameter tensor. Together they give the total trainable parameter count — with a caveat for weight-tied parameters being counted twice.

---

## What Is a Parameter?

A parameter is a tensor that the optimizer updates during training. In PyTorch, parameters are created with `nn.Parameter`:

```python
self.scale = nn.Parameter(torch.ones(768))   # trainable ✓
self.mask  = torch.ones(256, 256)             # plain tensor — not a parameter ✗
```

Only `nn.Parameter` objects are tracked by `nn.Module` and updated by the optimizer.

---

## model.parameters()

`model.parameters()` recursively walks the entire model hierarchy and yields every `nn.Parameter` it finds. For GPT-2, it yields:

```
GPT2Model.parameters() yields:
  tok_emb.weight              (50,257 × 768)
  pos_emb.weight              (   256 × 768)

  For each of 12 transformer blocks:
    ln1.scale, ln1.shift
    att.W_query.weight, att.W_query.bias
    att.W_key.weight,   att.W_key.bias
    att.W_value.weight, att.W_value.bias
    att.W_out.weight,   att.W_out.bias
    ln2.scale, ln2.shift
    ff.layers[0].weight, ff.layers[0].bias
    ff.layers[2].weight, ff.layers[2].bias

  final_norm.scale, final_norm.shift
  out_head.weight             (50,257 × 768)  ← same tensor as tok_emb.weight if tied
```

---

## p.numel()

`numel()` returns the **total number of scalar elements** in a tensor:

```python
p = torch.ones(768, 3072)
p.numel()   # → 2,359,296  (768 × 3072 = 2,359,296 individual numbers)

p = torch.ones(50257, 768)
p.numel()   # → 38,597,376  (50,257 × 768)
```

---

## Counting Total Parameters

```python
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
# → 162,000,000  (without weight tying)
# →  124,000,000  (with weight tying)
```

This iterates through every parameter tensor, counts its elements, and sums them all up.

---

## The Weight Tying Caveat

With weight tying, `out_head.weight` and `tok_emb.weight` point to the **same tensor in memory**. But `model.parameters()` yields both names — it does not check for duplicates. So the 50,257×768 = 38.6M parameter tensor is counted twice.

```python
# Standard count — counts tied params twice
sum(p.numel() for p in model.parameters())        # → ~162M

# Deduplicated count — correct unique count
sum(p.numel() for p in set(model.parameters()))   # → ~124M
```

`set()` removes duplicates based on object identity — if two names point to the same tensor, it only counts it once.

---

## What Is and Is Not Included

| Tensor type | Included in parameters()? | Trained by optimizer? |
|---|---|---|
| `nn.Parameter` (weights, biases) | Yes | Yes |
| `register_buffer` (causal_mask) | No | No |
| Plain Python attributes (n_heads=12) | No | No |
| Weight-tied tensor | Yes — counted twice | Yes — same tensor updated once |

`register_buffer` tensors are saved in `model.state_dict()` (for checkpointing) but excluded from `model.parameters()` (not trained). They are a middle category — PyTorch-aware but not learnable.

---

## Practical Example

```python
model = GPT2Model(GPT_CONFIG_124M)

# Count all parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total: {total:,}")   # 162,086,400 (without weight tying)

# Count unique parameters
unique = sum(p.numel() for p in set(model.parameters()))
print(f"Unique: {unique:,}") # 123,488,768 (with weight tying)

# Count only trainable parameters (same as all, since no frozen layers here)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}")
```

---

## Related

- [[weight-tying]]
- [[gpt2-from-scratch]]
- [[gpt2-parameter-count]]
- [[register-buffer]]
