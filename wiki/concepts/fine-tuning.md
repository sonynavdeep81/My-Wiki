---
title: Fine-Tuning
type: concept
tags: [fine-tuning, instruction-tuning, classification, LoRA, PEFT, transfer-learning]
sources: 1
updated: 2026-04-13
---

## Fine-Tuning

**Summary**: Additional training of a pre-trained (foundational) LLM on a smaller, domain-specific dataset to adapt it for a particular task or behavior.

## Two Main Types

### Instruction Fine-Tuning

Trains the model to follow natural language instructions across diverse tasks (e.g., "Summarize this:", "Translate to French:").

- Requires large, diverse instruction datasets
- Updates the **entire model** over long sequences
- High compute and memory demand
- Produces general-purpose chat/assistant models (e.g., InstructGPT, ChatGPT)

### Classification Fine-Tuning

Replaces the LM output head with a classification head (e.g., 2-class output for spam/ham). Trains on labeled examples for a specific task.

- Smaller dataset sufficient
- Lower compute — often only the head (and sometimes a few top layers) are trained
- Produces task-specific models

**Example**: SMS spam classification on UCI dataset
- Balanced dataset (equal spam/ham samples)
- Labels: spam=1, ham=0
- Split: 70% train / 10% val / 20% test

## PEFT: Parameter-Efficient Fine-Tuning

Instruction fine-tuning's high compute cost led to **PEFT** — freeze most model weights and train only a small set of additional parameters.

### LoRA (Low-Rank Adaptation)

Injects trainable low-rank matrices into attention layers alongside frozen original weights:
```
W_updated = W_frozen + A × B   (where A: d×r, B: r×d, r << d)
```
Only A and B are trained — a tiny fraction of total parameters.

### QLoRA

Combines LoRA with **4-bit quantization** of the frozen base model weights. Further reduces memory usage, enabling fine-tuning of large models on consumer GPUs.

| Method | Trainable Params | Memory | Use Case |
|---|---|---|---|
| Full fine-tuning | 100% | High | Instruction tuning with resources |
| LoRA | ~0.1–1% | Medium | Instruction or classification |
| QLoRA | ~0.1–1% | Low | Large models on limited hardware |

PEFT is most valuable for instruction fine-tuning but can also be applied to classification when the base model is large.

## Related

- [[large-language-models]]
- [[scaling-laws]]
- [[gpt2-from-scratch]]
- [[decoder-only-architecture]]
- [[llama]]
