---
title: Instruction Fine-Tuning — Data Format (Instruction + Desired Response)
type: query
tags: [fine-tuning, instruction-tuning, dataset, training-data]
sources: 1
updated: 2026-04-30
---

## Instruction Fine-Tuning — Data Format (Instruction + Desired Response)

**Summary**: Instruction fine-tuning trains on (instruction, desired response) pairs — the model sees both sides during training and learns to generate correct free-form outputs.

## Training Data Format

| Instruction (input) | Desired Response (target) |
|---|---|
| Convert 45 kilometers to meters. | 45 kilometers is 45000 meters. |
| Provide a synonym for "bright". | A synonym for "bright" is "radiant". |
| Edit the following sentence to remove all passive voice: "The song was composed by the artist." | The artist composed the song. |

Both columns are fed to the model during training. The loss is computed on the response tokens only — the instruction is context, not a prediction target.

## Why Both Sides Are Needed

The model must learn:
- **What** the instruction is asking (task type)
- **How** a correct response looks (output format, style, content)

Seeing only the instruction gives no gradient signal about what "correct" looks like. Seeing only the response gives no context. The pair together is what shapes the behavior.

## Contrast: Classification Fine-Tuning

| Aspect | Instruction Fine-Tuning | Classification Fine-Tuning |
|---|---|---|
| Target output | Free-form generated text | Discrete label (e.g. spam=1) |
| Training pair | (instruction, response text) | (input text, class id) |
| Dataset size needed | Large + diverse | Smaller |
| Compute | High | Lower |
| Output head | Same LM head (vocab-size) | Replaced with num_classes head |

## Related

- [[fine-tuning]]
- [[gpt2-from-scratch]]
- [[large-language-models]]
- [[instruction-finetuning-prompt-format]]
