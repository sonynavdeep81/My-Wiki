---
title: Inference Scaling
type: concept
tags: [inference, scaling, reasoning, 2025]
sources: 1
updated: 2026-04-13
---

## Inference Scaling

**Summary**: A 2025 paradigm shift where model capability is improved by giving models more compute at inference time (longer "thinking"), rather than training ever-larger models.

## The Shift

Prior to 2025, capability gains came primarily from **parameter scaling** — training larger models. In 2025, the industry shifted focus to **inference scaling**: spending more compute per query to improve output quality.

Mechanisms include:
- Chain-of-thought / extended reasoning before answering
- Search over multiple candidate answers
- Iterative self-correction
- Models that "think" for longer before responding (e.g., GPT-5.1 Thinking, ~2T parameter estimate)

## Why Now

[[scaling-laws]] for pretraining are showing diminishing returns at the frontier. Inference scaling provides a complementary axis: instead of bigger models, smarter use of compute per query.

## Relation to [[kv-caching]]

Inference scaling increases the number of tokens generated per query, which amplifies the importance of KV Caching for efficiency.

## Related

- [[scaling-laws]]
- [[emergent-abilities]]
- [[large-language-models]]
- [[kv-caching]]
