---
title: LLM Evaluation
type: concept
tags: [evaluation, benchmarks, mmlu, instruction-fine-tuning]
sources: 1
updated: 2026-05-07
verified_against: instruction_fine_tuning, 2026-05-07
confidence: high
---

## LLM Evaluation

**Summary**: Instruction-finetuned LLMs are evaluated via benchmarks, human preference, or automated LLM scoring — not simple label matching.

## Why Standard Accuracy Doesn't Work

Classification tasks: compare predicted label vs true label → accuracy.
Instruction-following: no single correct answer → needs richer evaluation.

## Three Evaluation Approaches

| Method | Example | Pro | Con |
|---|---|---|---|
| Benchmark tests | MMLU | Objective, scalable | Narrow; doesn't test open-ended quality |
| Human preference comparison | LMSYS Chatbot Arena | Reflects real user preference | Expensive, slow |
| Automated LLM scoring | AlpacaEval (GPT-4 judge) | Scalable, no humans | Inherits judge model's biases |

## MMLU (Measuring Massive Multitask Language Understanding)

- 57 subjects: STEM, humanities, social sciences, law, medicine, etc.
- Difficulty: elementary → professional level
- Format: multiple choice (A/B/C/D) → easy to score automatically
- Settings: zero-shot and few-shot
- Designed because older benchmarks were too easy for modern LLMs

## Fair Model Comparison

- Always compare models of **similar size**
- Fine-tuned GPT-2 (355M) → baseline: original GPT-2 (355M)
- Comparing to GPT-3 (175B) is unfair — parameter count differs by 500×
- Goal of comparison: show what fine-tuning added, not absolute capability

## Practical Guidance

- Learning/practice projects: manual inspection of responses is sufficient
- Research papers: run MMLU on fine-tuned vs base model; report delta
- Production systems: combine all three approaches

## Related

- [[Instruction Fine-Tuning]]
- [[Perplexity]]
- [[Fine-Tuning]]
