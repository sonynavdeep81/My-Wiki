---
title: Label Smoothing
type: concept
tags: [regularization, label-smoothing, training, classification]
sources: 1
updated: 2026-04-14
---

## Label Smoothing

**Summary**: A regularization technique that replaces hard 0/1 training targets with soft targets, distributing a small probability mass across all classes to prevent overconfident predictions.

---

## The Problem With Hard Targets

Standard cross-entropy trains the model toward a one-hot target:

```
Hard target for correct class "cat" (class 2 of 5):
[0, 0, 1, 0, 0]
```

This pushes the logit for the correct class toward +∞ and all others toward −∞. The model becomes **overconfident** — it never learns to be uncertain, which hurts generalization.

---

## How Label Smoothing Works

With smoothing ε, the target distribution becomes:

```
Smoothed target (ε=0.1, 5 classes):
[0.02, 0.02, 0.92, 0.02, 0.02]

Formula:
  correct class:  1 - ε + ε/K  =  1 - 0.1 + 0.02  =  0.92
  other classes:  ε/K           =  0.1/5            =  0.02
```

where K = number of classes.

The model is now trained to assign a small nonzero probability to every class — it can never be infinitely confident.

---

## Effect on Training

| Metric | Effect |
|---|---|
| Perplexity | Gets **worse** — model is intentionally less certain |
| Accuracy / BLEU | Gets **better** — better generalization, less overfitting |
| Calibration | Improves — predicted probabilities more closely reflect true confidence |

The paper states: *"Label smoothing hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."*

---

## In the Attention Is All You Need Paper

- ε = 0.1
- Applied during training on the translation task
- Contributed to the SOTA BLEU scores alongside dropout

**Not used in your GPT-2 implementation** (not noted in the GPT-2 notebook).

---

## Related

- [[dropout]]
- [[Attention_2023|Attention Is All You Need (Paper)]]
- [[attention-is-all-you-need]]
- [[decoding-strategies]]
