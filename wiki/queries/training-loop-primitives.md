---
title: Training Loop Primitives — train/eval/zero_grad/backward/step/no_grad
type: query
tags: [pytorch, training, optimization, backpropagation, dropout]
sources: 1
updated: 2026-05-14
---

## Training Loop Primitives — train/eval/zero_grad/backward/step/no_grad

**Summary**: The six essential PyTorch calls in a training loop, what each one does, and why the order matters.

---

## Overview

A typical training loop in PyTorch looks like this:

```python
model.train()                           # set training mode

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()           # clear old gradients
        outputs = model(inputs)         # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # compute new gradients
        optimizer.step()                # update weights

    model.eval()                        # set eval mode for validation
    with torch.no_grad():
        val_loss = compute_val_loss(val_loader, model)
    model.train()                       # back to training mode
```

---

## model.train() and model.eval()

These two calls switch the model between training mode and evaluation mode.

| Call | Dropout | BatchNorm |
|---|---|---|
| `model.train()` | Active — randomly zeros activations | Uses current batch statistics |
| `model.eval()` | Disabled — all values pass through | Uses running average statistics |

**Why it matters:** Dropout behaves differently in training vs inference. During training it randomly drops activations to prevent overfitting. During inference you want deterministic, full-strength predictions — so dropout must be off.

**Correct placement:**
- `model.train()` — before the training loop starts
- `model.eval()` — before validation/inference
- `model.train()` — again after validation, before the next training epoch

---

## optimizer.zero_grad()

```python
optimizer.zero_grad()
```

PyTorch **accumulates** gradients by default — each call to `loss.backward()` adds to the existing `.grad` values rather than replacing them. If you don't clear them at the start of each batch, gradients from previous batches pile up and corrupt your updates.

**Must be called before `loss.backward()`**, once per batch.

---

## Forward Pass — Building the Computation Graph

```python
outputs = model(inputs)
loss = criterion(outputs, labels)
```

When you run the forward pass in training mode, PyTorch builds a **computation graph** — a record of every operation performed, storing intermediate values needed to compute gradients later. This graph is what makes automatic differentiation possible.

---

## loss.backward() — Computing Gradients

```python
loss.backward()
```

Walks backward through the computation graph using the chain rule and computes `∂loss/∂param` for every parameter with `requires_grad=True`. These gradients are stored in `param.grad`.

**Important:** `backward()` only computes gradients. It does NOT update any weights. That is the optimizer's job.

---

## optimizer.step() — Updating Weights

```python
optimizer.step()
```

Reads the `.grad` value for every parameter and applies the update rule. For AdamW:

```
param = param - lr × AdamW_update(param.grad, momentum_state)
```

**Important:** `step()` only updates weights. It does NOT compute gradients. That is `backward()`'s job.

**Why are they separate?** So you can do things between gradient computation and weight update — most importantly, gradient clipping:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
optimizer.step()
```

Gradient clipping prevents very large gradients from causing unstable weight updates (exploding gradients problem).

---

## torch.no_grad() — Disabling the Computation Graph

```python
with torch.no_grad():
    val_outputs = model(val_inputs)
    val_loss = criterion(val_outputs, val_labels)
```

Even without calling `loss.backward()`, PyTorch builds the computation graph during every forward pass — just in case you might call `backward()` later. This costs memory and time.

`torch.no_grad()` tells PyTorch: "don't build the graph inside this block." Since you never backpropagate during validation, this is pure waste that can be skipped.

**Benefits:**
- Reduces memory usage (no intermediate values stored)
- Speeds up forward pass

**Rule:** Always wrap validation and inference loops in `torch.no_grad()`. Never wrap the training loop in it.

---

## The Correct Order — Why It Matters

```
optimizer.zero_grad()   ← must come before backward()
loss = forward(...)     ← builds computation graph
loss.backward()         ← computes gradients, stores in .grad
                        ← optional: clip_grad_norm_() here
optimizer.step()        ← reads .grad, updates weights
```

If you swap `zero_grad()` and `backward()`, gradients from the previous batch accumulate into the current batch's gradients. The weight updates become wrong — the model "remembers" past gradients incorrectly.

---

## Summary Table

| Call | Does | Does NOT do |
|---|---|---|
| `model.train()` | Enables dropout, batch stats | Update weights |
| `model.eval()` | Disables dropout | Update weights |
| `zero_grad()` | Clears `.grad` on all params | Compute or apply gradients |
| `loss.backward()` | Computes gradients → stores in `.grad` | Update weights |
| `optimizer.step()` | Updates weights using `.grad` | Compute gradients |
| `torch.no_grad()` | Disables graph construction | Freeze parameters |

---

## Related

- [[dropout]]
- [[optimizer]]
- [[gpt2-from-scratch]]
- [[why-save-optimizer-state]]
