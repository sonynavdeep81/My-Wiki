---
title: Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)
type: query
tags: [fine-tuning, instruction-tuning, loss-masking, padding, next-token-prediction, training]
sources: 1
updated: 2026-05-06
---

## Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)

**Summary**: Instruction fine-tuning uses dynamic per-batch padding, next-token prediction (target = input shifted by 1), and loss masking on padding tokens. Note: instruction token masking is a recommended best practice but is NOT applied in the reference implementation — instruction tokens contribute to loss.

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

## 3. Loss Masking — What This Implementation Actually Masks

```
Sequence:  [### Instruction: Convert... ### Response: 45 km is 45000 m. <eos> <pad> <pad>]
Loss mask: [  ✓   ✓    ✓      ✓         ✓    ✓          ✓   ✓   ✓   ✓    ✓     ✗     ✗  ]
```

**This implementation (custom_collate) masks padding tokens only:**
- Instruction tokens → loss computed normally (contribute to weight updates)
- Response tokens → loss computed normally
- First `<eos>` (50256) → kept as real prediction target
- Padding `<eos>` beyond first → masked to **-100** (cross-entropy ignores)

**Recommended best practice (not applied here):** also mask instruction tokens so the model is only graded on responses. This gives a cleaner instruction-following signal but requires tracking the instruction length per sample.

```
Ideal mask: [  ✗   ✗    ✗      ✗         ✗    ✗          ✓   ✓   ✓   ✓    ✓     ✗     ✗  ]
```

## 4. What `### Response:` Does

- Acts as a **trigger token** at inference: you feed the full prompt including `### Response:` and the model continues with the answer
- The model learned "after this delimiter, generate an answer" from context, not from loss
- Remove it at inference → model loses the boundary signal → generates garbage or starts responding mid-instruction

## 5. Contrast with Classification FT

| Aspect | Classification FT | Instruction FT |
|---|---|---|
| Padding | Global max (all batches same length) | Dynamic per-batch |
| Output target | Single class label (integer) | Tokens shifted by 1 |
| Loss computed on | Output head only (last token position) | All tokens except padding (instruction NOT masked in reference impl) |
| Mechanics | Discrimination | Next-token prediction (same as pretraining) |

## Related

- [[fine-tuning]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[stanford-alpaca]]
- [[dropout-during-finetuning]]
- [[training-loop-primitives]]
