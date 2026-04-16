---
title: model.parameters() and p.numel() Explained
type: query
tags: [pytorch, nn.Module, parameters, numel, weight-tying, gpt2]
updated: 2026-04-14
---

## `model.parameters()` and `p.numel()` Explained

**Summary**: `model.parameters()` recursively yields every `nn.Parameter` in the model hierarchy; `p.numel()` returns the total scalar element count of a tensor. Together they count total trainable parameters — with a caveat for weight-tied parameters being counted twice.

---

## `model.parameters()`

Recursively walks every `nn.Module` in the hierarchy and yields every `nn.Parameter` it finds. In your GPT-2 implementation:

```
GPT2Model.parameters() yields:
  tok_emb.weight              (50,257 × 768)
  pos_emb.weight              (256    × 768)
  trf_blocks[0..11] each:
    ln1.weight, ln1.bias
    att.W_query.weight/bias
    att.W_key.weight/bias
    att.W_value.weight/bias
    att.W_out.weight/bias
    ln2.weight, ln2.bias
    ff.layers[0].weight/bias
    ff.layers[2].weight/bias
  final_norm.weight, final_norm.bias
  out_head.weight             (50,257 × 768) ← same tensor as tok_emb.weight
```

---

## `p.numel()`

Returns the total number of scalar elements in a tensor:

```python
p = torch.ones(768, 3072)
p.numel()  # → 2,359,296  (768 × 3072)
```

The full expression:

```python
sum(p.numel() for p in model.parameters())
```

Iterates every parameter, counts elements, sums — giving total trainable scalar values.

---

## Weight Tying Caveat

`out_head.weight` and `tok_emb.weight` point to the **same tensor** in memory (see [[weight-tying]]). `model.parameters()` yields both, so the sum counts 50,257×768 = 38.6M parameters **twice**.

To get the true unique count:

```python
# Counts duplicates (what print usually shows)
sum(p.numel() for p in model.parameters())

# Deduplicated count
sum(p.numel() for p in set(model.parameters()))
```

---

## What Is and Isn't Included

| Tensor type | In `parameters()`? | Trained? |
|---|---|---|
| `nn.Parameter` (weights, biases) | Yes | Yes |
| `register_buffer` (causal_mask) | No | No |
| Plain Python attributes (n_heads) | No | No |
| Tied weights (out_head = tok_emb) | Yes — counted twice | Same tensor |

`register_buffer` tensors are excluded because they are non-trainable — they appear in `model.state_dict()` but not in `model.parameters()`. See [[register-buffer|Why Use register_buffer?]].

---

## Related

- [[pytorch-nn-building-blocks|PyTorch nn Building Blocks]]
- [[weight-tying]]
- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[register-buffer|Why Use register_buffer?]]
