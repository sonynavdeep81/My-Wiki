---
title: Why the Inference Template Must Match the Training Template
type: query
tags: [instruction-fine-tuning, alpaca, inference, prompt-template]
updated: 2026-05-18
---

## Why the Inference Template Must Match the Training Template

**Summary**: The model does not understand "instructions" in general — it learned to respond to one specific pattern of tokens. Use a different pattern at inference and the model gets confused.

---

## The Core Idea

During fine-tuning, every training example is wrapped in the same template — for example, the Alpaca format:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the following sentence to French.

### Input:
The weather is nice today.

### Response:
Le temps est agréable aujourd'hui.
```

After seeing thousands of examples in this exact pattern, the model learns one reflex:

> *"Whenever I see `### Response:`, that is my cue to start generating the answer."*

---

## Training vs Inference — Side by Side

**Training time** (model sees input + output together):

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the following sentence to French.

### Input:
The weather is nice today.

### Response:
Le temps est agréable aujourd'hui.
```

**Inference time** (you feed everything up to `### Response:`, then let the model complete):

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the following sentence to French.

### Input:
The sky is clear tonight.

### Response:
```

The model sees `### Response:` — its learned trigger fires — and it generates:

```
Le ciel est clair ce soir.
```

---

## What Happens If You Break the Pattern

Suppose at inference you use a slightly different template:

```
Instruction: Translate the following sentence to French.
Input: The sky is clear tonight.
Answer:
```

Two things are wrong:
- `### Instruction:` became `Instruction:` — the `###` tokens are missing
- `### Response:` became `Answer:` — the trigger token is completely different

The model never saw `Answer:` during training. Its learned reflex does not fire. It may repeat the instruction, generate nonsense, or produce an empty response.

---

## The Dog Analogy

You train a dog with the command *"Sit!"* — always that exact word.

At test time you say *"Please be seated."* — same meaning, different words. The dog stares at you.

The dog did not learn the concept of sitting-on-command. It learned the specific sound *"Sit!"*

The fine-tuned model is the same. It did not learn "instructions" in the abstract. It learned the specific token sequence `### Response:` as the signal to generate output. Change the signal, break the behavior.

---

## The Non-Negotiable Rule

> Whatever template wraps the training data must be used **identically** at inference.

This means:
- Same header text (`Below is an instruction...`)
- Same section markers (`### Instruction:`, `### Input:`, `### Response:`)
- Same handling of empty `input` (skip the `### Input:` block entirely if empty — don't leave it blank)

The template is not decoration. It **is** the language the fine-tuned model speaks.

---

## Quick Reference (Colab Notes)

- Wrap every training example in the template before tokenizing
- At inference, fill in `instruction` + `input`, include `### Response:` but leave it empty
- The model was not taught "follow instructions" — it was taught "complete this exact token pattern"
- Typo in the delimiter = broken model, even if the meaning is identical

---

## Related

- [[Instruction Fine-Tuning]]
- [[instruction-finetuning-prompt-format]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-data-pipeline]]
