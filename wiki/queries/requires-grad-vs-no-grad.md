---
title: requires_grad vs torch.no_grad()
type: query
tags: [pytorch, gradients, freezing, inference, fine-tuning, training]
sources: 0
updated: 2026-04-24
---

## requires_grad vs torch.no_grad()

**Summary**: `requires_grad=False` permanently freezes specific parameters during training; `torch.no_grad()` temporarily suspends all gradient tracking for inference/validation.

## Comparison

| | `requires_grad=False` | `torch.no_grad()` |
|---|---|---|
| Scope | Specific parameters | All operations in block |
| Duration | Permanent until changed | Temporary |
| Purpose | Freeze layers during training | Speed up inference/validation |
| Saves memory? | No | Yes |
| Blocks weight update? | Indirectly (no gradient exists) | N/A — optimizer never called at inference |

## Weight Update Nuance

Neither explicitly blocks the optimizer. Both work by ensuring **no gradient exists** — so the optimizer has nothing to apply:

- `requires_grad=False` → gradient never computed → optimizer step has no effect on this param
- `torch.no_grad()` → used at inference where optimizer is never called anyway

If `optimizer.step()` were accidentally called on a frozen param, it wouldn't update it — but only because `.grad` is `None`, not because the optimizer knows it's frozen.

## requires_grad=False — Freeze Specific Parameters

```python
# Step 1 — freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 2 — unfreeze only what you want to train
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
for param in model.out_head.parameters():
    param.requires_grad = True
```

- Permanent setting until explicitly changed back
- PyTorch skips gradient computation for frozen params entirely
- Standard first step in classification fine-tuning strategy

## torch.no_grad() — Suspend All Gradient Tracking

```python
with torch.no_grad():
    output = model(x)   # no computation graph built, no gradients stored
```

- Temporary — only applies inside the block
- Disables graph construction → saves memory + speeds up forward pass
- Does NOT affect which parameters get updated
- Wrap validation/inference loops; never wrap training loop

## Related

- [[fine-tuning]]
- [[classification-finetuning-strategy]]
- [[training-loop-primitives]]
- [[optimizer]]
