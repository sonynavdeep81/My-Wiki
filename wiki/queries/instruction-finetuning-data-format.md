---
title: Instruction Fine-Tuning — Data Format (Instruction + Desired Response)
type: query
tags: [fine-tuning, instruction-tuning, dataset, training-data]
sources: 1
updated: 2026-05-14
---

## Instruction Fine-Tuning — Data Format (Instruction + Desired Response)

**Summary**: Instruction fine-tuning trains on (instruction, desired response) pairs. The model sees both sides during training and learns to generate correct free-form outputs. Loss is computed only on response tokens.

---

## What Does the Training Data Look Like?

Each training example is a pair: an instruction and the ideal response to that instruction.

| Instruction | Desired Response |
|---|---|
| Convert 45 kilometers to meters. | 45 kilometers is 45,000 meters. |
| Provide a synonym for "bright". | A synonym for "bright" is "radiant". |
| Edit the following sentence to remove passive voice: "The song was composed by the artist." | The artist composed the song. |

The instructions can be anything — math conversions, grammar corrections, question answering, summarization, coding tasks. The more diverse the dataset, the more capable the resulting model.

---

## Why Does the Model See Both Sides During Training?

During training, the full (instruction + response) text is fed to the model as one continuous sequence. The model reads the instruction as context and then is trained to predict the response tokens.

**Why the instruction is needed:** Without the instruction, the model has no context for what kind of response is appropriate. It cannot learn that "Convert 45 km to meters" should produce a unit conversion answer rather than a poem.

**Why the response is needed:** The response provides the training signal. The model's loss is computed on how well it predicts each response token. Without the response, there is nothing to learn from.

Together, the pair teaches the model: "when you see this kind of instruction, this is what a correct response looks like."

---

## Loss Is Computed Only on Response Tokens

Although the model reads both instruction and response, the loss (and therefore the weight updates) is computed **only on response tokens**. Instruction tokens are ignored during loss calculation.

This makes sense: the instruction is given information — the model is not being graded on its ability to reproduce the question. It is being graded on its ability to produce a correct answer.

---

## How This Differs from Classification Fine-Tuning

| Aspect | Instruction Fine-Tuning | Classification Fine-Tuning |
|---|---|---|
| Target output | Free-form generated text | Discrete label (e.g. spam=1) |
| Training pair | (instruction text, response text) | (input text, class ID) |
| Output head | Same LM head (50,257 classes) | Replaced with num_classes head |
| Dataset size | Large and diverse needed | Smaller datasets work |
| Compute | Higher | Lower |
| Task type | Open-ended generation | Fixed-label prediction |

Classification fine-tuning produces one number (a class). Instruction fine-tuning produces a sequence of tokens — an entire response. The underlying mechanism (next-token prediction) is the same as pretraining, just applied to instruction-response pairs.

---

## Related

- [[fine-tuning]]
- [[gpt2-from-scratch]]
- [[instruction-finetuning-prompt-format]]
- [[instruction-finetuning-data-pipeline]]
