---
title: Instruction Fine-Tuning — Why Only Targets Are Masked (Not Inputs)
type: query
tags: [fine-tuning, instruction-tuning, loss-masking, targets, inputs, cross-entropy]
updated: 2026-05-18
---

## Instruction Fine-Tuning — Why Only Targets Are Masked (Not Inputs)

**Summary**: Only targets are masked with -100 because only targets are used to compute the loss. Inputs are never compared against anything — they just go into the model.

---

## The Two Roles

| | inputs | targets |
|---|---|---|
| What they are | Token IDs fed into the model | Token IDs the model must predict |
| Used for | Forward pass | Loss computation |
| Need masking? | No | Yes |

---

## Why inputs are never masked

Inputs go into the model's forward pass. The model processes every token — it doesn't matter whether a position is padding or real content. There is no comparison, no loss, nothing to skip. Putting `-100` in inputs would just confuse the model with an invalid token ID.

---

## Why targets are masked

Loss is computed as:

```python
cross_entropy(logits, targets)
```

If a target position contains padding (50256), it carries no meaningful signal — the model should not be penalized for failing to predict a padding token. Replacing those positions with `-100` tells PyTorch's `cross_entropy` to skip them entirely via `ignore_index=-100`.

---

## The flow

```
inputs  →  model  →  logits
                         ↓
targets  →  cross_entropy(logits, targets)  ←  -100 positions skipped here
```

The model never sees targets. Masking only ever happens on the right side of that arrow.

---

## One-line rule

> `-100` is not a "hide this token" signal for the model — it is a "skip this position" signal for the loss function.

---

## What Happens at -100 Positions — The Full Chain

- During the forward pass, the model **still produces predictions (logits)** at every position, including `-100` positions
- `cross_entropy` with `ignore_index=-100` sees those positions and **completely skips** them — loss is not calculated
- Because loss is not calculated → gradient is not calculated
- Because gradient is not calculated → weights are not updated
- These positions are **totally invisible** to the learning process

**What -100 does NOT mean:**
- It does not mean loss = 0 (loss is not computed at all, not computed as zero)
- It does not mean gradient = 0 (gradient is skipped entirely, not zeroed out)

**Which positions get -100 in targets:**
- All padding tokens beyond the first stop token (50256)
- The first stop token is kept as a real prediction target — the model must learn to emit it

---

## Related

- [[instruction-fine-tuning]]
- [[instruction-finetuning-training-mechanics]]
- [[cross-entropy-loss]]
- [[instruction-finetuning-dataset-creation-steps]]
