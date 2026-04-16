---
title: PyTorch nn Building Blocks
type: concept
tags: [pytorch, nn.Module, nn.Linear, nn.Embedding, nn.Parameter, nn.Sequential]
sources: 0
updated: 2026-04-14
---

## PyTorch nn Building Blocks

**Summary**: `nn.Module` is the base class that gives PyTorch classes learnable-weight tracking, device management, and checkpoint I/O; `nn.Parameter`, `nn.Linear`, `nn.Embedding`, and `nn.Sequential` are the four core primitives built on top of it.

---

## Why inherit from `nn.Module`?

When a class inherits from `nn.Module`, PyTorch automatically provides:

| Capability | Why it matters |
|---|---|
| `.parameters()` iterator | Optimizer finds all learnable weights |
| `.state_dict()` / `.load_state_dict()` | Save/load checkpoints |
| `.to(device)` / `.train()` / `.eval()` | Propagates recursively to all sub-modules |
| Auto-registration of sub-modules | Child layers are tracked without manual bookkeeping |
| Hook system, `.zero_grad()` | Gradient management and introspection |

Without `nn.Module`, PyTorch has no way to discover that a tensor is a learnable weight — the optimizer would silently skip it.

---

## `nn.Parameter`

A **raw tensor wrapper** that marks a tensor as a learnable weight.

```python
self.scale = nn.Parameter(torch.ones(768))
```

- `requires_grad=True` by default
- Auto-registered when assigned as a module attribute
- Use when: you need a **custom weight** not covered by a built-in layer
  - Examples: [[Layer Normalization|LayerNorm]]'s γ and β, RoPE rotation frequencies, any learned scalar

---

## `nn.Linear`

A **fully-connected layer**: `y = xW^T + b`

```python
self.proj = nn.Linear(768, 3072)  # W: (3072, 768), b: (3072,)
```

- Input: float tensor of shape `(..., in_features)`
- Output: float tensor of shape `(..., out_features)`
- Internally holds two `nn.Parameter`s: `weight` and `bias`
- Use when: projecting continuous feature vectors — Q/K/V projections in [[Multi-Head Attention]], [[feed-forward-network|FFN]] layers, output classification head

---

## `nn.Embedding`

A **lookup table**: integer index → dense vector

```python
self.tok_emb = nn.Embedding(50257, 768)  # (vocab_size, d_model)
```

- Input: integer tensor of token (or position) indices
- Output: float tensor — rows selected from the weight matrix
- Forward pass is pure indexing (`weight[idx]`), not a matrix multiply
- Use when: converting discrete tokens or positions into continuous vectors
- See [[Embeddings]] and [[Weight Tying]] (GPT-2 shares this matrix with the output head)

**vs `nn.Linear`**: `nn.Embedding` takes `int` indices and does table lookup; `nn.Linear` takes `float` tensors and does `xW^T`.

---

## `nn.Sequential`

An **ordered container** that chains modules, piping each output into the next input.

```python
self.ff = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768),
)
```

- No custom `forward()` needed — just list the layers in order
- Use when: the forward pass is a **simple pipeline** with no branching or intermediate reuse
- Cannot express: skip/residual connections, attention (needs separate Q/K/V), anything needing intermediate values
- Works perfectly for the [[feed-forward-network|FFN]] sub-block; a custom `forward()` is needed for [[Multi-Head Attention]]

---

## Mental Model

```
nn.Module            ← skeleton: registry, device mgmt, save/load
  ├── nn.Parameter   ← raw learnable tensor (custom weights)
  ├── nn.Linear      ← float → float via matrix multiply
  ├── nn.Embedding   ← int index → float vector via lookup table
  └── nn.Sequential  ← ordered pipeline container
```

---

## Related

- [[Embeddings]]
- [[Multi-Head Attention]]
- [[feed-forward-network|Feed-Forward Network (FFN)]]
- [[Layer Normalization]]
- [[Weight Tying]]
- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
