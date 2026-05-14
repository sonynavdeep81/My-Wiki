---
title: requires_grad vs torch.no_grad()
type: query
tags: [pytorch, gradients, freezing, inference, fine-tuning, training]
sources: 0
updated: 2026-05-14
---

## requires_grad vs torch.no_grad()

**Summary**: `requires_grad=False` permanently freezes specific parameters so they are never updated during training. `torch.no_grad()` temporarily suspends all gradient tracking for a block of code — used for inference and validation to save memory and speed up the forward pass.

---

## The Core Difference

These two tools solve different problems:

- **`requires_grad=False`** — controls *which parameters* get updated during training
- **`torch.no_grad()`** — controls *when* gradient tracking happens at all

---

## requires_grad=False — Freezing Parameters

Every `nn.Parameter` has a `requires_grad` flag. When `True` (the default), PyTorch tracks how the loss depends on that parameter and computes its gradient during backpropagation. When `False`, PyTorch skips it entirely.

**Typical fine-tuning pattern — freeze everything, then selectively unfreeze:**

```python
# Step 1 — freeze all parameters
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

**What happens to frozen parameters:**
- No gradient is computed for them during `loss.backward()`
- `param.grad` stays `None`
- When `optimizer.step()` runs, it finds no gradient → makes no update
- The parameter value stays exactly the same throughout training

**Duration:** permanent until you explicitly set `requires_grad = True` again.

---

## torch.no_grad() — Suspending Gradient Tracking

Even when you are not calling `optimizer.step()`, PyTorch normally builds a computation graph during every forward pass. This graph stores intermediate values needed to compute gradients later — which costs memory and time.

`torch.no_grad()` tells PyTorch: "don't build a computation graph for anything inside this block."

```python
# Validation loop — correct usage
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)     # no graph built, no gradients stored
        loss = criterion(outputs, labels)
        # loss is just a number — no backprop possible or needed
```

**What torch.no_grad() does:**
- Disables computation graph construction
- Reduces memory usage (no intermediates stored)
- Speeds up the forward pass
- Does NOT affect which parameters would be updated — the optimizer is simply never called in this block

**Duration:** temporary — only applies inside the `with` block.

---

## Side-by-Side Comparison

| | `requires_grad=False` | `torch.no_grad()` |
|---|---|---|
| Scope | Specific parameters | All operations in the block |
| Duration | Permanent until changed | Temporary (only inside `with` block) |
| Purpose | Freeze layers during training | Speed up inference/validation |
| Saves memory? | No | Yes |
| Affects optimizer? | Indirectly (no gradient to apply) | N/A — optimizer not called in this context |
| Typical use | Fine-tuning: freeze pretrained layers | Validation loop, inference |

---

## An Important Nuance — Neither "Blocks" the Optimizer Directly

Neither mechanism explicitly prevents the optimizer from running. They both work by ensuring **no gradient exists**:

- `requires_grad=False` → gradient is never computed → `param.grad` is `None` → optimizer has nothing to apply
- `torch.no_grad()` → gradient is never computed → used in contexts where optimizer is never called anyway

If you accidentally called `optimizer.step()` inside `torch.no_grad()`, or on a frozen parameter, no update would happen — but only because `.grad` is `None`, not because of any direct block.

---

## When to Use Which

**Use `requires_grad=False` when:**
- Fine-tuning: you want most of the model frozen, training only a small subset of layers
- You want the freeze to persist across many forward/backward passes

**Use `torch.no_grad()` when:**
- Running validation during training (wrap the entire val loop)
- Running inference at test time
- Computing embeddings or features without training
- Any time you do a forward pass but don't need backpropagation

---

## Related

- [[fine-tuning]]
- [[classification-finetuning-strategy]]
- [[training-loop-primitives]]
- [[optimizer]]
