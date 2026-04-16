---
title: Why Save the Optimizer State in a Checkpoint?
type: query
tags: [pytorch, optimizer, training, checkpointing, adamw]
sources: 1
updated: 2026-04-16
---

## Why Save the Optimizer State in a Checkpoint?

**Summary**: The optimizer has its own internal state (momentum, adaptive learning rates) separate from model weights — discarding it on resume causes erratic training.

---

## What the Optimizer Remembers

For [[optimizer|AdamW]], it tracks two extra values **per parameter**:

- **m** (1st moment) — running average of gradients. *"Which direction have I been moving recently?"*
- **v** (2nd moment) — running average of squared gradients. *"How bumpy has this path been?"*

It also tracks a **step counter** — how many updates have happened so far (used for bias correction in Adam).

---

## Why This Matters

Adam doesn't just use the current gradient to update a weight. It uses the **history** of gradients. After thousands of batches, m and v have accumulated meaningful momentum.

If you discard them on resume:

```
Training stops at batch 5000 → m and v built up over 5000 steps
Resume without saving → m=0, v=0 again
Model weights are fine, but optimizer "forgets" all momentum
→ First few hundred batches behave erratically, loss spikes
```

If you restore them:

```
Resume → m and v exactly as they were at batch 5000
→ Training continues smoothly as if it never stopped
```

---

## Analogy

Think of the model weights as a student's **knowledge** (what they've learned). The optimizer state is their **study rhythm** — knowing which topics they've been struggling with, which ones are going smoothly, and how aggressively to study each. Saving only the knowledge but forgetting the rhythm means they have to rediscover their study habits from scratch.

---

## What's Inside `checkpoint.pth`

```python
torch.save({
    'model_state_dict':     model.state_dict(),      # all weights & biases
    'optimizer_state_dict': optimizer.state_dict(),  # m, v, step count per param
}, 'checkpoint.pth')
```

Loading it back:

```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## Summary

| What's saved | Contains | Why needed |
|---|---|---|
| `model_state_dict` | All weights & biases | What the model knows |
| `optimizer_state_dict` | m, v, step count per param | How it was learning |

**Model weights = what the model knows. Optimizer state = how it was learning.** Both are needed to resume training properly.

---

## Related

- [[optimizer]]
- [[gpt2-from-scratch]]
- [[training-loop-primitives]]
