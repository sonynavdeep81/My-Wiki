---
title: Why optimizer.zero_grad() Is Needed
type: query
tags: [pytorch, optimizer, gradients, training, backpropagation]
sources: 1
updated: 2026-05-14
---

## Why optimizer.zero_grad() Is Needed

**Summary**: PyTorch accumulates (adds) gradients across backward passes rather than replacing them. `optimizer.zero_grad()` clears the accumulated gradients before each new batch so that weight updates reflect only the current batch, not a corrupted mix of all previous batches.

---

## The Problem: PyTorch Accumulates Gradients

In PyTorch, every time you call `.backward()`, it does **not replace** the old gradients — it **adds** the new gradients on top of whatever was already there.

This is intentional behavior (useful in some advanced cases like gradient accumulation), but during normal training it causes a serious bug if you forget to clear them.

---

## A Simple Analogy

Imagine you're a teacher grading papers. After each student's exam, you write down their score on a running tally sheet.

- **Wrong approach**: You never erase the sheet. By student 3, the score shows the sum of all three students' scores — not just student 3's.
- **Right approach**: Before grading each new exam, you wipe the sheet clean. Each student gets their own fresh score.

`optimizer.zero_grad()` is the "wipe the sheet clean" step — done before each new batch.

---

## What Happens Without It

```python
# BAD — no zero_grad
for batch in dataloader:
    loss = model(batch)
    loss.backward()        # gradients ADDED to previous batch's gradients
    optimizer.step()       # updates weights using wrong, accumulated gradients
```

After batch 1: gradients = g1
After batch 2: gradients = g1 + g2  ← wrong! Should be just g2
After batch 3: gradients = g1 + g2 + g3 ← keeps growing

The model updates its weights using a corrupted gradient that mixes all previous batches together. Training becomes unstable or diverges.

---

## What Happens With It

```python
# GOOD — zero_grad before each batch
for batch in dataloader:
    optimizer.zero_grad()  # wipe gradients from last batch
    loss = model(batch)
    loss.backward()        # compute fresh gradients for THIS batch only
    optimizer.step()       # update weights using correct gradients
```

Each batch starts with a clean slate. The gradient used to update the weights reflects only the current batch — exactly what you want.

---

## The Correct Order (Every Training Step)

```
1. optimizer.zero_grad()   → clear old gradients
2. output = model(input)   → forward pass
3. loss = criterion(...)   → compute loss
4. loss.backward()         → compute new gradients
5. optimizer.step()        → update weights
```

Never swap steps 1 and 4 — zeroing after backward throws away the gradients you just computed.

---

## When Accumulation Is Intentional

There is one case where you deliberately skip `zero_grad()` for several batches — **gradient accumulation** — used when your GPU cannot fit a large batch. You accumulate gradients over N small batches to simulate 1 large batch, then call `optimizer.step()` and `optimizer.zero_grad()` once.

```python
# Gradient accumulation over 4 steps (simulates 4x larger batch)
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()                          # accumulates for 4 steps

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                     # update once after 4 batches
        optimizer.zero_grad()                # then clear
```

Even here, zero_grad is called — just less frequently. For normal training, always zero_grad before every batch.

---

## Related

- [[training-loop-primitives]] — full training loop: zero_grad, backward, step, train/eval, no_grad
- [[optimizer]] — AdamW internals, momentum, adaptive learning rates
- [[requires-grad-vs-no-grad]] — when gradient tracking is suspended entirely
