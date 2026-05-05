---
title: Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)
type: query
tags: [fine-tuning, instruction-tuning, loss-masking, padding, next-token-prediction, training]
sources: 1
updated: 2026-04-30
---

## Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)

**Summary**: Instruction fine-tuning uses dynamic per-batch padding, next-token prediction (target = input shifted by 1), and loss masking so the model only learns from response tokens.

## 1. Dynamic vs Static Padding

| | Classification FT | Instruction FT |
|---|---|---|
| Padding strategy | Global `max_tokens` from entire train set | Per-batch: pad to longest sequence in that batch |
| All batches same length? | Yes | No — each batch width varies |
| Why? | Small fixed inputs (emails) | Instruction inputs vary wildly in length |

## 2. Target = Input Shifted by 1

Instruction fine-tuning is still **next-token prediction** — identical mechanics to pretraining.

```
Full sequence:  [### Instruction: Convert... ### Response: 45 km is 45000 m. <eos>]
Input to model: [### Instruction: Convert... ### Response: 45 km is 45000 m.      ]  (positions 0..N-1)
Target:         [    Instruction: Convert... ### Response: 45 km is 45000 m. <eos>]  (positions 1..N)
```

At every position the model predicts "what token comes next."

## 3. Loss Masking — Only Learn from the Response

```
Sequence:  [### Instruction: Convert... ### Response: 45 km is 45000 m. <eos>]
Loss mask: [  ✗   ✗    ✗      ✗         ✗    ✗          ✓   ✓   ✓   ✓    ✓  ]
```

- Instruction + delimiter tokens → label set to **-100** (PyTorch cross-entropy ignores this index)
- Response tokens → loss computed normally, weights updated
- Padding tokens → also masked (-100)

**Analogy:** fill-in-the-blank exam. The model reads the full question but only gets graded on what it writes in the answer box. Marking the model on the question text would teach it the wrong thing.

**Why not mask the delimiter `### Response:` itself?**
It is usually masked too. But the model still learns the delimiter pattern from seeing it thousands of times as *input context* — no loss needed for that.

## 4. What `### Response:` Does

- Acts as a **trigger token** at inference: you feed the full prompt including `### Response:` and the model continues with the answer
- The model learned "after this delimiter, generate an answer" from context, not from loss
- Remove it at inference → model loses the boundary signal → generates garbage or starts responding mid-instruction

## 5. Contrast with Classification FT

| Aspect | Classification FT | Instruction FT |
|---|---|---|
| Padding | Global max (all batches same length) | Dynamic per-batch |
| Output target | Single class label (integer) | Tokens shifted by 1 |
| Loss computed on | Output head only (last token position) | Response tokens only (instruction masked) |
| Mechanics | Discrimination | Next-token prediction (same as pretraining) |

## Related

- [[fine-tuning]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[stanford-alpaca]]
- [[dropout-during-finetuning]]
- [[training-loop-primitives]]
