---
title: Training Loop Primitives — train/eval/zero_grad/backward/step/no_grad
type: query
tags: [pytorch, training, optimization, backpropagation, dropout]
sources: 1
updated: 2026-04-16
---

## Training Loop Primitives

**Summary**: What each PyTorch training loop call does, explained in simple terms for beginners.

---

## `model.train()` and `model.eval()`

Think of this like a **student switching between study mode and exam mode**.

- In **study mode** (`model.train()`), the student deliberately makes things harder for themselves — they practice with some information randomly hidden (that's [[dropout]]). This forces them to not rely on any single piece of information, making them more robust.
- In **exam mode** (`model.eval()`), nothing is hidden. The student uses everything they know to give their best answer.

The model itself doesn't learn anything when you call these — it's just telling the model *how to behave* for what's coming next.

| Layer | `model.train()` | `model.eval()` |
|---|---|---|
| [[dropout]] | Randomly switches off some neurons | All neurons active |
| BatchNorm | Uses current batch's statistics | Uses stored statistics |

**Why placed where they are:**
- `model.train()` — called before training starts, so dropout is active during learning.
- `model.eval()` — called before validation, so the model gives its best deterministic answer.
- `model.train()` again — called after validation to switch back for the next epoch.

---

## `optimizer.zero_grad()`

Imagine you're keeping score on a whiteboard. After each round, you **erase the board** before writing the new score. That's `zero_grad()`.

PyTorch has a quirk: every time you call `.backward()`, it **adds** the new gradients on top of whatever was already there — it doesn't replace them. So if you forget to erase, you're making decisions based on a mix of old and new information.

```
batch 1: whiteboard shows g1
batch 2 (forgot to erase): whiteboard shows g1 + g2  ← wrong, stale info
batch 2 (erased first):    whiteboard shows g2        ← correct, fresh info
```

**Always call it before `loss.backward()`** so each batch starts with a clean slate.

---

## `loss.backward()`

This is where the actual **backpropagation** happens.

After the model makes a prediction, you calculate the loss (how wrong it was). `loss.backward()` then asks: *"For each weight in the model, if I nudge it slightly, does the loss go up or down — and by how much?"*

That "how much" is called a **gradient**, and it gets stored inside each parameter as `param.grad`.

Important: **`.backward()` only calculates and stores gradients. It does not change any weights.**

---

## `optimizer.step()`

Now that `.backward()` has figured out which direction each weight should move, `optimizer.step()` actually **moves them**.

It reads the gradients stored in `param.grad` and updates every weight accordingly:

```
new_weight = old_weight - (learning_rate × gradient)
```

Think of it like this:
- `.backward()` is the GPS calculating the route.
- `optimizer.step()` is actually driving the car.

They are intentionally separate steps. This lets you do things in between — like gradient clipping (capping extreme gradients before applying them) — without changing the backprop logic.

---

## `torch.no_grad()`

When the model does a forward pass, PyTorch secretly **saves all intermediate calculations** in the background in case you later call `.backward()` (it needs them to compute gradients).

During validation, you never call `.backward()` — you're just checking performance, not learning. So all that behind-the-scenes bookkeeping is pure waste: it uses memory and slows things down for no reason.

`torch.no_grad()` says: *"Don't bother saving anything — we're just looking, not learning."*

```python
with torch.no_grad():   # no bookkeeping, saves memory
    val_loss = cal_loader_loss(val_loader, model)
```

---

## Full Flow Per Batch — The Big Picture

```
1. zero_grad()       → erase last batch's gradients (clean whiteboard)
2. forward pass      → model makes predictions, loss is calculated
                       (PyTorch quietly saves intermediate values for step 3)
3. loss.backward()   → backprop: compute gradients, store in param.grad
                       (GPS calculates the route)
4. optimizer.step()  → update all weights using param.grad
                       (car drives along the route)
```

Steps 3 and 4 together are what people mean by "the model is learning."

---

## Related

- [[dropout]]
- [[optimizer]]
- [[gpt2-from-scratch]]
