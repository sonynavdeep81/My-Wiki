---
title: Scaling Laws
type: concept
tags: [scaling, training, compute, parameters, data]
sources: 1
updated: 2026-04-13
---

## Scaling Laws

**Summary**: Empirical relationships showing that LLM performance improves predictably as parameters, dataset size, and compute (FLOPs) are scaled up together.

## The Three Axes

Model performance scales with:
1. **Parameters** — number of weights in the model
2. **Dataset size** — tokens seen during training
3. **Compute (FLOPs)** — total floating point operations during training

Scaling any one of these in isolation yields diminishing returns; optimal scaling balances all three (Chinchilla scaling laws).

## Connection to Emergent Abilities

[[emergent-abilities]] are closely tied to scaling laws. Once a model reaches a specific threshold across parameters, data, and compute, accuracy on certain tasks **jumps suddenly from near-zero to significant levels**. The jump is non-linear — it doesn't gradually improve.

## 2025 Shift

By 2025, the industry shifted from pure parameter scaling to **[[inference-scaling]]** — giving models more compute at inference time (e.g., chain-of-thought, search, longer "thinking"). See [[inference-scaling]].

## GPT-3 Training Scale

- 300 billion training tokens
- Training data mix: Common Crawl (60%), WebText2 (22%), Books (16%), Wikipedia (3%)
- Training cost: ~$4.6 million

## Related

- [[emergent-abilities]]
- [[inference-scaling]]
- [[large-language-models]]
