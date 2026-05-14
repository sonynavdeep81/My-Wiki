---
title: Why Save the Optimizer State in a Checkpoint?
type: query
tags: [pytorch, optimizer, training, checkpointing, adamw]
sources: 1
updated: 2026-05-14
---

## Why Save the Optimizer State in a Checkpoint?

**Summary**: The optimizer's internal state (momentum, velocity, step count) is separate from the model's weights. Discarding it when resuming training causes loss spikes and slow recovery — the optimizer behaves as if training is starting from scratch.

---

## What Gets Saved in a Checkpoint?

A training checkpoint typically contains two things:

```python
torch.save({
    'model_state_dict':     model.state_dict(),      # learned weights and biases
    'optimizer_state_dict': optimizer.state_dict(),  # optimizer's internal state
}, 'checkpoint.pth')
```

Most people understand why model weights are saved — you need them to resume training or run inference. But why save the optimizer state?

---

## What Is the Optimizer State?

For AdamW (the most common optimizer for LLMs), each parameter has three values stored in the optimizer state:

| Value | Symbol | What it is |
|---|---|---|
| First moment | m | Running average of past gradients (tracks direction of updates) |
| Second moment | v | Running average of squared gradients (tracks per-parameter scale) |
| Step count | t | Number of updates applied (used for bias correction) |

These values build up over many training steps. AdamW uses them to compute adaptive learning rates — parameters that have received large gradients recently get smaller updates, and vice versa.

---

## What Happens If You Don't Save the Optimizer State?

If you save only the model weights and then resume training:

```python
# Resuming without optimizer state:
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = AdamW(model.parameters(), lr=4e-4)  # fresh optimizer — m=0, v=0, t=0
```

The optimizer starts as if training just began:
- `m = 0` — no direction history
- `v = 0` — no scale history
- `t = 0` — bias correction assumes step 1

For the first hundreds of batches, the optimizer makes poor update decisions because it has no momentum built up. The loss spikes noticeably and takes time to recover — even though the model weights themselves are perfectly correct.

---

## The Correct Resume Pattern

```python
# Save:
torch.save({
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')

# Resume:
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Training continues exactly as if it never stopped
```

With the optimizer state restored, `m`, `v`, and `t` are exactly where they were when training was interrupted. The optimizer picks up in the middle of its learned trajectory — no spike, no warm-up period needed.

---

## An Analogy

Think of the optimizer state as muscle memory for a runner.

- **Model weights** = the runner's current physical fitness (how fast they can run right now)
- **Optimizer state** = their training rhythm, pacing strategy, and fatigue patterns built up over months

If you save the fitness but reset the muscle memory, the runner starts their next session with their full physical capability but clumsy form — stumbling for a few minutes before finding their rhythm again. Save both and they resume with perfect form from the first step.

---

## When Is It Okay NOT to Save Optimizer State?

- **Transfer learning from a published checkpoint:** you intentionally want a fresh optimizer since you're fine-tuning with different data and possibly different learning rates
- **Changing the optimizer type or learning rate:** the saved state is incompatible anyway
- **Short training runs:** if training completes in one session without interruption, no checkpoint is needed mid-training

---

## Related

- [[optimizer]]
- [[gpt2-from-scratch]]
- [[training-loop-primitives]]
