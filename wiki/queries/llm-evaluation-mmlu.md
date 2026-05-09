---
title: How do we evaluate LLMs? (MMLU & comparison strategy)
type: query
tags: [evaluation, mmlu, benchmarks, instruction-fine-tuning]
updated: 2026-05-07
---

## How do we evaluate LLMs?

**Summary**: Unlike classification, instruction-finetuned LLMs have no single correct answer — evaluation requires benchmarks, human preference, or LLM scoring.

See [[LLM Evaluation]] for full details on all three approaches, MMLU structure, and fair model comparison guidelines.

**Key takeaway filed here**: For a practice GPT-2 fine-tuning project, manual response inspection is sufficient. MMLU is for research; fair comparison is always same-size models (GPT-2 355M vs GPT-2 355M, not GPT-3).

## Related

- [[LLM Evaluation]]
- [[Instruction Fine-Tuning]]
- [[Perplexity]]
