---
title: Training Loop Primitives — train/eval/zero_grad/backward/step/no_grad
type: query
tags: [pytorch, training, optimization, backpropagation, dropout]
sources: 1
updated: 2026-04-17
---

## Training Loop Primitives

**Summary**: Roles of each PyTorch training loop call; backward/step division of labor.

## Mode Switches

| Call | Effect on [[dropout]] | Effect on BatchNorm |
|---|---|---|
| `model.train()` | Active (random zeroing) | Uses current batch stats |
| `model.eval()` | Disabled | Uses running stats |

Placement: `train()` before epoch loop; `eval()` before val; `train()` again after val.

## Per-Batch Flow

```
optimizer.zero_grad()   # clear param.grad — PyTorch accumulates by default
loss = forward(...)     # builds computation graph
loss.backward()         # backprop: ∂loss/∂param → stored in param.grad
optimizer.step()        # param -= lr * f(param.grad); reads grad, updates weights
```

- `zero_grad` must precede `backward` — else gradients from prior batch accumulate
- `backward` computes gradients only; does **not** update weights
- `optimizer.step` updates weights only; does **not** compute gradients
- Separation allows gradient clipping between the two: `clip_grad_norm_()` after `backward`, before `step`

## `torch.no_grad()`

- Forward pass builds computation graph (stores intermediates for backprop) even if `.backward()` never called
- `no_grad()` disables graph construction → saves memory + compute
- Wrap val loop; never wrap train loop

```python
with torch.no_grad():
    val_loss = cal_loader_loss(val_loader, model)
```

## Related

- [[dropout]]
- [[optimizer]]
- [[gpt2-from-scratch]]
