---
title: Learning Rate Warmup
type: concept
tags: [training, optimizer, learning-rate, warmup, schedule]
sources: 1
updated: 2026-04-18
---

## Learning Rate Warmup

**Summary**: Gradually increases learning rate from 0 (or small value) to the target LR over the first N steps before applying the main schedule.

## Why It's Needed

At init, weights are random — gradients are large and noisy. A high LR at step 0 causes destructive updates before the model has any signal. Warmup keeps early updates small until the loss landscape becomes navigable.

Without warmup: early loss spikes, instability, sometimes divergence (especially in large models with Pre-LN).

## Common Schedule: Linear Warmup + Cosine Decay

```
step < warmup_steps:  lr = target_lr × (step / warmup_steps)
step >= warmup_steps: lr = cosine decay from target_lr to min_lr
```

```python
# PyTorch example
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.0004, total_steps=total_steps,
    pct_start=0.1   # 10% of steps = warmup
)
```

## Typical Values

| Model size | Warmup steps |
|-----------|-------------|
| Small (GPT-2 demo) | 100–500 |
| Medium (GPT-2 124M) | 2000 |
| Large (GPT-3) | 375M tokens |

## GPT-2 Notebook

The notebook (GPT2_Clean) uses a fixed lr=0.0004 with AdamW — **no warmup scheduler**. Warmup is recommended for longer training runs or larger models.

## Related

- [[optimizer]]
- [[gpt2-from-scratch]]
- [[scaling-laws]]
