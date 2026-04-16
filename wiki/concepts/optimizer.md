---
title: Adam and AdamW Optimizers
type: concept
tags: [optimizer, adam, adamw, training, pytorch]
sources: 2
updated: 2026-04-14
---

## Adam and AdamW Optimizers

**Summary**: Adaptive gradient optimizers that maintain per-parameter learning rates using estimates of first and second gradient moments; AdamW fixes a weight decay bug in Adam and is the standard for LLM training.

---

## Adam

Proposed by Kingma & Ba (2015). Maintains two moving averages per parameter:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t        # 1st moment (mean of gradients)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²       # 2nd moment (variance of gradients)

m̂_t = m_t / (1 - β₁ᵗ)                      # bias correction
v̂_t = v_t / (1 - β₂ᵗ)

θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)
```

Parameters with consistently large gradients get a smaller effective learning rate; rare parameters get a larger one. This makes it well-suited for sparse gradients (e.g. embeddings).

---

## AdamW — Weight Decay Fix

Standard Adam implements L2 regularization by adding `λ·θ` to the gradient before the update. This couples weight decay with the adaptive learning rate, making the effective decay inconsistent across parameters.

AdamW decouples them — weight decay is applied **directly** to the weights, separately from the gradient step:

```python
# Adam L2 (incorrect coupling):
g_t = g_t + λ · θ          # decay folded into gradient
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)

# AdamW (decoupled):
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)   # gradient step
θ_t = θ_t - lr · λ · θ_{t-1}               # weight decay applied separately
```

---

## Configurations

### In your GPT-2 implementation
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
```
- Fixed learning rate (no warmup schedule)
- weight_decay=0.1

### In the Attention Is All You Need paper
```python
# Adam (not AdamW)
# β₁=0.9, β₂=0.98, ε=10⁻⁹
# Warmup + inverse-sqrt decay schedule:
lrate = d_model^-0.5 · min(step^-0.5, step · warmup_steps^-1.5)
# warmup_steps = 4000
```
- Custom LR schedule: linear warmup for 4000 steps, then decay
- No weight decay noted

---

## Why AdamW Over Adam for LLMs

- Correct weight decay behavior — proven to improve generalization
- Better regularization without distorting adaptive learning rates
- Default in `transformers` library (Hugging Face) for all model training
- GPT-2, GPT-3, LLaMA all trained with AdamW or its variants

---

## Related

- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[Attention_2023|Attention Is All You Need (Paper)]]
- [[dropout]]
- [[fine-tuning]]
