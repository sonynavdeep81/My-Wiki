---
title: How Do We Evaluate LLMs? (MMLU and Comparison Strategy)
type: query
tags: [evaluation, mmlu, benchmarks, instruction-fine-tuning]
updated: 2026-05-14
---

## How Do We Evaluate LLMs? (MMLU and Comparison Strategy)

**Summary**: Unlike classification, instruction-fine-tuned LLMs have no single correct answer — evaluation requires benchmarks, human preference scoring, or LLM-as-judge. For a practice project, manual inspection is sufficient.

---

## The Core Problem

When you fine-tune a model for classification, evaluation is simple: did it predict the right label? Correct or incorrect. Easy to measure.

When you fine-tune a model for instruction following, evaluation is hard: the model generates free-form text. "The artist composed the song" and "The song was written by the artist" are both correct answers to the same instruction. There is no single ground truth to compare against.

Three main strategies have emerged to handle this.

---

## Strategy 1 — Benchmarks (e.g., MMLU)

**MMLU (Massive Multitask Language Understanding)** is a multiple-choice benchmark covering 57 subjects — mathematics, law, medicine, history, computer science, and more. Because it is multiple-choice, there is a definite correct answer for every question.

**How it works:**
```
Question: "What is the powerhouse of the cell?"
Options: (A) Nucleus  (B) Mitochondria  (C) Ribosome  (D) Golgi apparatus
Correct: (B)
```

The model is scored on how often it selects the correct option. MMLU score is reported as accuracy across all 57 subjects.

**What MMLU measures:** broad knowledge and reasoning across academic domains.

**What it doesn't measure:** whether the model can follow instructions well, write clearly, be helpful in conversation, or avoid harmful outputs.

---

## Strategy 2 — Human Preference Evaluation

Human evaluators are shown outputs from two different models (or two versions of the same model) and asked: "Which response is better?"

The result is reported as a **win rate** — what percentage of comparisons did Model A win over Model B.

**Advantage:** captures qualities that benchmarks miss — helpfulness, tone, clarity, safety.

**Disadvantage:** slow, expensive, and requires careful design to avoid evaluator bias.

---

## Strategy 3 — LLM-as-Judge

A powerful LLM (e.g., GPT-4) is used to evaluate the outputs of the model being tested. The judge LLM is given the instruction and the model's response, and asked to rate quality on a scale or pick the better of two responses.

**Advantage:** much faster and cheaper than human evaluation; scales to thousands of examples.

**Disadvantage:** the judge LLM has its own biases and may prefer outputs that sound like its own style, regardless of correctness.

---

## Fair Comparison Rules

When comparing two models using any of these strategies, comparisons must be controlled:

- **Same model size:** GPT-2 355M vs GPT-2 355M. Comparing GPT-2 124M to GPT-3 (175B) is not informative — the size difference overwhelms everything else.
- **Same training data:** unless you are specifically studying the effect of data
- **Same evaluation prompt format:** a model trained with Alpaca format must be evaluated with Alpaca format
- **Same decoding settings:** temperature, top-k, etc.

Violating any of these makes the comparison meaningless.

---

## For a Practice GPT-2 Fine-Tuning Project

For learning purposes, manual inspection is perfectly sufficient:

1. Generate 10–20 responses to held-out instructions
2. Read them and ask: does the model understand what is being asked?
3. Does the response format match what was requested?
4. Are there obvious errors — wrong facts, wrong format, repetition?

MMLU and LLM-as-judge are for research papers and production systems. For a single-person learning project, your own judgment is the right evaluation tool.

---

## Related

- [[LLM Evaluation]]
- [[Instruction Fine-Tuning]]
- [[Perplexity]]
- [[llm-evaluation-metrics]]
