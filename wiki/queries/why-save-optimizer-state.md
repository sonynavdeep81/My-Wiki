---
title: Why Save the Optimizer State in a Checkpoint?
type: query
tags: [pytorch, optimizer, training, checkpointing, adamw]
sources: 1
updated: 2026-04-17
---

## Why Save the Optimizer State in a Checkpoint?

**Summary**: Optimizer state (m, v, step) is separate from model weights; discarding it on resume causes loss spikes until momentum rebuilds.

## What `optimizer.state_dict()` Contains (AdamW)

Per parameter:
- **m** — 1st moment: running average of gradients (direction)
- **v** — 2nd moment: running average of squared gradients (per-param scale)
- **step** — update count (used for Adam bias correction)

## Why It Matters

AdamW weight update uses history, not just current gradient. If state is lost on resume:
- m=0, v=0 → optimizer behaves as if training just started
- Loss spikes for hundreds of batches until momentum rebuilds
- Model weights are correct but adaptive learning rates are wrong

## Checkpoint Pattern

```python
# Save
torch.save({
    'model_state_dict':     model.state_dict(),      # weights & biases
    'optimizer_state_dict': optimizer.state_dict(),  # m, v, step per param
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

| Key | Contains |
|---|---|
| `model_state_dict` | All `nn.Parameter` tensors |
| `optimizer_state_dict` | m, v, step per param + hyperparams |

## Related

- [[optimizer]]
- [[gpt2-from-scratch]]
- [[training-loop-primitives]]
