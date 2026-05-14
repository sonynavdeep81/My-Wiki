---
title: Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)
type: query
tags: [fine-tuning, instruction-tuning, loss-masking, padding, next-token-prediction, training]
sources: 1
updated: 2026-05-14
---

## Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)

**Summary**: Instruction fine-tuning uses dynamic per-batch padding, next-token prediction (target = input shifted by 1), and loss masking on padding tokens. The reference implementation does NOT mask instruction tokens — they contribute to loss. Masking instructions is a best practice but requires extra implementation effort.

---

## The Core Mechanic — Still Next-Token Prediction

Instruction fine-tuning uses exactly the same training objective as pretraining: **next-token prediction**. At every position, the model predicts the next token. The only difference is that the training data is now instruction-response pairs instead of raw text.

```
Full sequence:  [### Instruction: Convert 45 km... ### Response: 45 km is 45000 m. <eos>]
                 ←─────────────────────────────────────────────────────────────────────→
Input to model: [### Instruction: Convert 45 km... ### Response: 45 km is 45000 m.      ]  positions 0..N-1
Target:         [    Instruction: Convert 45 km... ### Response: 45 km is 45000 m. <eos>]  positions 1..N
```

The target is the input shifted left by one position. For every input token, the model is asked: "what token comes next?"

---

## Dynamic Padding vs Static Padding

Classification fine-tuning typically pads all sequences to a single global maximum length (e.g., the longest email in the entire spam dataset). This works because emails have similar lengths.

Instruction fine-tuning uses **dynamic per-batch padding** — each batch is padded only to the length of its longest sequence:

```
Batch 1: longest = 120 tokens → all sequences padded to 120
Batch 2: longest = 340 tokens → all sequences padded to 340
```

**Why dynamic?** Instruction lengths vary enormously — a single-word task ("Summarize:") vs a long paragraph translation task. Static global padding would waste massive amounts of memory and computation on short sequences padded to the length of the longest possible instruction.

| | Classification FT | Instruction FT |
|---|---|---|
| Padding strategy | Global max (entire dataset) | Per-batch max |
| All batches same width? | Yes | No |
| Memory efficiency | Lower | Higher |

---

## Loss Masking — What Gets Masked

Not every token in the sequence should contribute equally to the loss.

**Padding tokens must be masked.** They are not real data — they are just filler to make the batch rectangular. If the model is penalized for not predicting padding tokens correctly, it would learn nonsense.

```
Sequence:  [### Instruction: Convert... ### Response: 45 km is 45000 m. <eos> <pad> <pad>]
Loss mask: [      ✓           ✓              ✓              ✓                ✓     ✗     ✗ ]
```

Padding tokens are replaced with `-100` in the target. PyTorch's `cross_entropy` automatically ignores any position with target `-100` (built-in `ignore_index`).

The **first `<eos>`** (50256) is kept as a real target — the model should learn to predict end-of-sequence. Only the padding `<eos>` tokens beyond the first are masked to `-100`.

---

## What This Implementation Does NOT Mask — Instruction Tokens

The ideal setup masks instruction tokens too — grading the model only on how well it generates the response, not on how well it reproduces the instruction:

```
Ideal mask:   [✗  ✗   ✗    ✗    ✗   ✗   ✗    ✓  ✓  ✓  ✓  ✓  ✗  ✗]
               ←── instruction ──→   ←── response ──→  ←padding→
```

**The reference implementation does NOT do this.** Instruction tokens contribute to the loss:

```
Actual mask:  [✓  ✓   ✓    ✓    ✓   ✓   ✓    ✓  ✓  ✓  ✓  ✓  ✗  ✗]
               ←── instruction ──→   ←── response ──→  ←padding→
```

**Why not mask instructions?** It requires tracking the exact token length of the instruction portion for each sample — extra bookkeeping in the collate function. The reference implementation skips this for simplicity.

**Does it matter?** In practice, not masking instructions usually still produces good results. The model is exposed to many more response tokens than instruction tokens across the full dataset. But masking instructions gives a cleaner gradient signal and is considered best practice in production systems.

---

## The Role of `### Response:` at Inference

During training, the model sees `### Response:` thousands of times followed by a correct answer. It learns: "after this delimiter, I should generate an answer."

At inference, you feed the full prompt including `### Response:` with no answer after it:
```
### Instruction: Convert 45 km to meters.
### Response:
```

The model continues from the `### Response:` token and generates the answer. If you remove this delimiter, the model loses the boundary signal and generates garbage.

---

## Comparison with Classification Fine-Tuning

| Aspect | Classification FT | Instruction FT |
|---|---|---|
| Padding | Global max, all batches same length | Dynamic per-batch |
| Target | Single class label (integer) | Token IDs shifted left by 1 |
| Loss on | Output head, last token position only | All non-padding tokens (+ instruction in reference impl) |
| Mechanics | Discrimination | Next-token prediction |
| Output | One number (class ID) | Full text sequence |

---

## Related

- [[fine-tuning]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[instruction-finetuning-data-pipeline]]
- [[dropout-during-finetuning]]
- [[training-loop-primitives]]
