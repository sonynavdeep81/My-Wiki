---
title: LayerNorm — Are Scale and Shift Shared Across Tokens?
type: query
tags: [layer-normalization, parameters, gpt-2, broadcasting]
sources: 2
updated: 2026-05-14
---

## LayerNorm — Are Scale and Shift Shared Across Tokens?

**Summary**: Yes. The learnable `scale (γ)` and `shift (β)` vectors are shared across every token in the sequence and every example in the batch. They are *per-feature*, not per-token. The normalization statistics (mean, variance) are computed *per token, fresh every time*; the affine transform that follows is the only learnable part — and it is global.

---

## The Question

In a Transformer block, LayerNorm is applied to every token's embedding. A natural confusion:

> "There are 1024 tokens in my sequence. Does each token have its own scale and shift, or do they share?"

The short answer: **they all share the same scale and shift**. Each of the 1024 tokens is normalized using the *same* learnable `(γ, β)` pair. But within a single token's 768-dim vector, every one of the 768 features has its own `γ` and `β` value.

---

## Shapes — GPT-2 124M Walkthrough

GPT-2 124M uses `emb_dim = 768` and `context_length = 1024`. Inside a Transformer block:

```
input  : (batch, seq_len, 768)     e.g. (2, 1024, 768)
γ scale: (768,)                    one value per feature
β shift: (768,)                    one value per feature
output : (batch, seq_len, 768)
```

Notice: `γ` and `β` have **only 768 entries each**. There is no `seq_len` dimension and no `batch` dimension. PyTorch broadcasts these vectors across the (batch, seq_len) axes automatically.

That means a LayerNorm layer in GPT-2 has only:

```
2 × 768 = 1536 parameters
```

regardless of sequence length, batch size, or how many tokens you push through it.

---

## What Happens, Step by Step

For a single token vector `x` of length 768:

**Step 1 — Compute statistics over its own 768 features:**

```
μ  = mean(x)           # scalar, computed over the 768 features of THIS token
σ² = var(x)            # scalar, computed over the same 768 features
```

These statistics are **per-token**. Every token in the sequence computes its own `μ` and `σ²` independently. Tokens do not share statistics with each other, and the batch is not mixed in.

**Step 2 — Normalize:**

```
x_hat = (x - μ) / sqrt(σ² + ε)     # shape (768,), mean ≈ 0, var ≈ 1
```

**Step 3 — Apply the shared affine:**

```
y = γ * x_hat + β                  # element-wise, both γ and β are (768,)
```

This is where `γ` and `β` come in. They are the **same vectors** for every token in the sequence and every example in the batch.

---

## Why This Design

### 1. Independence from sequence length

Per-token statistics mean that LayerNorm does not care whether you feed it 4 tokens or 1024 tokens. Each token is normalized using only its own 768 features. The same layer works at every position.

If statistics were shared across tokens (like BatchNorm shares across the batch), then the normalization at training time (long sequences, large batch) would not match the normalization at inference time (single token, batch=1).

### 2. Independence from batch size

Per-token statistics also mean LayerNorm gives the same output whether your batch size is 1 or 32. This is a major contrast with [[Layer Normalization]] vs BatchNorm:

| | Statistics over | Sensitive to batch size? | Inference behavior |
|---|---|---|---|
| BatchNorm | batch, height, width (per channel) | Yes | Uses running averages |
| LayerNorm | features (per token) | No | Same as training |

For autoregressive language models, batch sizes vary wildly between training and inference, so LayerNorm is the right fit.

### 3. Tiny parameter count

If `γ` and `β` were per-position (one pair for each of the 1024 positions), you would need:

```
2 × 1024 × 768 = 1,572,864 parameters per LayerNorm
```

GPT-2 124M has 25 LayerNorm layers (one per block ×2 + one final). That would balloon to ~39M parameters just for normalization — most of which would be wasted, since position-specific normalization rarely helps.

Instead, the actual cost is:

```
2 × 768 × 25 = 38,400 parameters
```

That's three orders of magnitude smaller.

### 4. Position-agnostic modeling

A subtle point: making `γ` and `β` per-position would *bake the sequence length into the model*. The model would only work for sequences of exactly 1024 tokens — you could not generalize to shorter sequences without padding tricks, and you could not extend to longer sequences at all. Sharing across positions keeps LayerNorm position-agnostic.

---

## Concrete Example

Suppose `seq_len = 3`, `emb_dim = 4`, batch=1:

```
x = [[[2.5, 4.8, 1.2, 5.5],     # token 0
      [0.0, 1.0, 2.0, 3.0],     # token 1
      [9.0, 9.0, 9.0, 9.0]]]    # token 2
```

**Per-token normalization:**

- Token 0: μ=3.5, σ≈1.71 → `x_hat₀ = [-0.58, 0.76, -1.34, 1.17]`
- Token 1: μ=1.5, σ≈1.12 → `x_hat₁ = [-1.34, -0.45, 0.45, 1.34]`
- Token 2: μ=9.0, σ=0    → `x_hat₂ = [0, 0, 0, 0]` (after ε)

Statistics differ per token. Token 2's statistics had nothing to do with Token 0's.

**Apply shared γ, β** (say `γ = [1.0, 1.0, 0.5, 2.0]`, `β = [0, 0, 1, 0]`):

```
y₀ = γ * x_hat₀ + β = [-0.58, 0.76, 0.33, 2.34]
y₁ = γ * x_hat₁ + β = [-1.34, -0.45, 1.23, 2.68]
y₂ = γ * x_hat₂ + β = [0, 0, 1, 0]
```

Same `γ` and `β` applied to all three tokens. The third feature always gets scaled by 0.5 and shifted by +1, regardless of which token it belongs to.

---

## PyTorch Reference

```python
nn.LayerNorm(normalized_shape=768, eps=1e-5, elementwise_affine=True)
```

- `normalized_shape=768` — normalize over the last dim of size 768.
- `elementwise_affine=True` — create the learnable `weight` (γ) and `bias` (β), each shape `(768,)`.
- Statistics are computed over the last dimensions matching `normalized_shape`, *for each token independently*.

To inspect the params:

```python
ln = nn.LayerNorm(768)
ln.weight.shape   # torch.Size([768])  ← γ
ln.bias.shape     # torch.Size([768])  ← β
sum(p.numel() for p in ln.parameters())  # 1536
```

---

## Common Misconceptions

| Misconception | Reality |
|---|---|
| "Each token learns its own γ, β" | No. γ, β are shared across all tokens. |
| "γ, β are scalars" | No. They are vectors of length `emb_dim`. One scalar per feature. |
| "LayerNorm has different params at different positions" | No. The same LayerNorm layer is applied identically at every position. |
| "Stats are computed across the batch" | No. Stats are computed *within* each token, never across batch or sequence. |
| "The model can learn to undo normalization completely" | Yes — when `γ = σ` and `β = μ`, the affine recovers the original `x`. This is why γ and β exist. |

---

## Related

- [[layer-normalization]] — the concept page (covers Pre-LN vs Post-LN, why normalization helps)
- [[gpt2-pretraining-implementation-notes]] — broader Q&A including LayerNorm code
- [[gpt2-parameter-count]] — full GPT-2 124M parameter breakdown including LayerNorm contribution
- [[model-parameters-numel]] — how PyTorch counts parameters across nested modules
