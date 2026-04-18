---
title: Temperature (Decoding)
type: concept
tags: [decoding, sampling, inference, temperature, generation]
sources: 1
updated: 2026-04-18
---

## Temperature (Decoding)

**Summary**: Scalar that divides logits before softmax to control the sharpness of the output probability distribution.

## Mechanism

```python
probs = softmax(logits / T)
```

Softmax is sensitive to *differences* between logits — temperature scales those differences up or down.

## Effect Table

| T | logits / T | Distribution after softmax |
|---|-----------|---------------------------|
| 1.0 | unchanged | original distribution |
| < 1 (e.g. 0.5) | amplified | peaks sharper; high-prob tokens dominate |
| > 1 (e.g. 2.0) | compressed | flatter; low-prob tokens get more chance |
| → 0 | → ±∞ | one-hot on argmax → greedy decoding |
| → ∞ | → 0 for all | uniform over vocabulary |

## Full Inference Pipeline

```
logits → top-k mask → logits / T → softmax → probs → multinomial → token
```

- **Temperature** controls *how confidently* to pick among candidates
- **Top-k** controls *which* tokens are candidates (applied before temperature)
- **Multinomial** is the actual random draw from the resulting distribution

See [[decoding-strategies]] for full pipeline code.

## Practical Values

| Task | Typical T |
|------|-----------|
| Code generation | 0.2–0.4 |
| Factual Q&A | 0.5–0.7 |
| Creative writing | 0.8–1.2 |
| Brainstorming / diversity | 1.2–1.5 |

## Related

- [[decoding-strategies]]
- [[gpt2-from-scratch]]
- [[large-language-models]]
