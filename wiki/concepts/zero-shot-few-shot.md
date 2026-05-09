---
title: Zero-Shot and Few-Shot Learning
type: concept
tags: [llm, prompting, generalization, in-context-learning, gpt]
sources: 1
updated: 2026-05-07
verified_against: Raschka-LLM-2025, 2026-05-07
confidence: high
---

## Zero-Shot and Few-Shot Learning

**Summary**: GPT-style LLMs generalize to unseen tasks from the prompt alone — zero-shot requires no examples, few-shot provides a small number of input-output demonstrations.

## Definitions

| Mode | Examples in prompt | How model learns |
|---|---|---|
| **Zero-shot** | 0 | Uses knowledge from pretraining only |
| **Few-shot** | 1–10 | Infers pattern from provided examples |
| **Fine-tuning** | Many (training set) | Weight updates via gradient descent |

Zero-shot and few-shot require no weight updates — both happen at inference time via prompt construction.

## Why LLMs Can Do This

Next-token pretraining on diverse internet text implicitly trains the model to recognize and continue patterns. Given a few examples of a task in the prompt, the model extends the pattern to the query. This is called **in-context learning** [well-established].

## GPT Behavior (Raschka Ch. 1)

```
# Zero-shot
Translate English to German:
breakfast =>            ← model completes with "Frühstück"

# Few-shot
goat => Ziege
shoe => Schuh
phone =>                ← model completes with "Telefon"
```

GPT models are also used for **text completion** — the most basic form: given partial text, continue it.

## Contrast with BERT

| | GPT (decoder) | BERT (encoder) |
|---|---|---|
| Training objective | Next-token prediction | Masked token prediction |
| Direction | Left-to-right (causal) | Bidirectional |
| Zero/few-shot | Strong | Weak |
| Classification | Via fine-tuning | Strong natively |

## Emergence

Zero-shot generalization is an [[emergent-abilities]] — it appears only above a certain model scale and was not an explicit training target.

## Related

- [[large-language-models]]
- [[emergent-abilities]]
- [[decoder-only-architecture]]
- [[instruction-fine-tuning]]
- [[bert]]
