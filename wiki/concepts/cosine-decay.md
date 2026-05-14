---
title: Cosine Decay (Learning Rate Schedule)
type: concept
tags: [training, optimizer, learning-rate, schedule, cosine, warmup]
sources: 1
updated: 2026-05-07
verified_against: Raschka-LLM-2025, 2026-05-07
confidence: high
---

## Cosine Decay

**Summary**: A learning rate schedule where the LR follows a half-cosine curve from peak down to near-zero after the warmup phase, preventing overshoot of loss minima in later training.

## Why It's Used

After warmup, keeping LR constant risks overshooting the loss minima. Cosine decay gradually slows weight updates as training converges, improving final model quality.

## Formula

$$
\text{progress} = \frac{\text{step} - \text{warmup\_steps}}{\text{total\_steps} - \text{warmup\_steps}}
$$

$$
\text{lr} = \text{min\_lr} + (\text{peak\_lr} - \text{min\_lr}) \cdot \tfrac{1}{2}\left(1 + \cos(\pi \cdot \text{progress})\right)
$$

- At $\text{progress} = 0$: $\text{lr} = \text{peak\_lr}$
- At $\text{progress} = 1$: $\text{lr} = \text{min\_lr}$
- Decay follows a half-cosine curve between those extremes

## Implementation (Raschka Appendix D)

```python
import math

if global_step < warmup_steps:
    lr = initial_lr + global_step * lr_increment      # linear warmup
else:
    progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
    lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

Typical values: `peak_lr=5e-4`, `initial_lr=3e-5`, `min_lr=1e-6`

## Full Schedule Shape

```
LR
|   /‾‾\
|  /    \___
| /         \__
|/              ‾‾‾
+--warmup--cosine decay-->  steps
```

## Always Paired With

| Technique | Role |
|---|---|
| [[lr-warmup]] | Phase 1: ramp up to peak_lr |
| Cosine decay | Phase 2: decay peak_lr → min_lr |
| [[gradient-clipping]] | Applied after warmup; prevents exploding gradients |

Together these three form the standard LLM training stabilization recipe (Appendix D of Raschka 2025).

## Related

- [[lr-warmup]]
- [[gradient-clipping]]
- [[optimizer]]
- [[gpt2-from-scratch]]
