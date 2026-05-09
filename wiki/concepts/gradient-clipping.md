---
title: Gradient Clipping
type: concept
tags: [training, gradients, stability, exploding-gradients, optimizer]
sources: 1
updated: 2026-05-07
verified_against: Raschka-LLM-2025, 2026-05-07
confidence: high
---

## Gradient Clipping

**Summary**: Caps gradient norm during backpropagation so parameter updates never exceed a maximum magnitude, preventing exploding gradients that destabilize LLM training.

## Problem It Solves

During training, gradient norms can spike — especially early when loss landscape is steep. A single bad batch can produce enormous gradients that blow up weights ("exploding gradient" problem).

## How It Works

Computes the L2 norm of all gradients in the model, then scales them down if the norm exceeds `max_norm`:

```
If ‖G‖₂ > max_norm:
    G' = G × (max_norm / ‖G‖₂)
```

Example: gradient matrix with L2 norm = 5, max_norm = 1 → scale by 1/5.

## PyTorch API

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Always called **after** `loss.backward()`, **before** `optimizer.step()`.

## Placement in Training Loop

```python
loss.backward()
if global_step > warmup_steps:                         # skip during warmup
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Raschka 2025 (Appendix D): clipping applied only *after* warmup phase — during warmup, LR is low enough that large gradients are less dangerous.

## Typical Values

| Setting | Value |
|---|---|
| max_norm | 1.0 (standard) |
| norm type | L2 (Euclidean) |
| Apply during warmup? | No (Raschka); Yes (some frameworks) |

## Related

- [[lr-warmup]]
- [[cosine-decay]]
- [[optimizer]]
- [[gpt2-from-scratch]]
