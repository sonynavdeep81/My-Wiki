---
title: Instruction Fine-Tuning — Data Preparation Pipeline (5 Steps)
type: query
tags: [fine-tuning, instruction-tuning, padding, loss-masking, tokenization, target-shift]
sources: 1
updated: 2026-05-14
---

## Instruction Fine-Tuning — Data Preparation Pipeline (5 Steps)

**Summary**: Raw instruction-response pairs go through 5 steps before reaching the model: format → tokenize → pad → shift to create targets → mask padding tokens in targets with -100.

---

## Why a Pipeline Is Needed

Unlike classification fine-tuning where each input is a short text and the target is a single number, instruction fine-tuning requires:
- A consistent text format wrapping each example
- Variable-length sequences padded to a uniform length per batch
- A target sequence that is the input shifted by one position
- Padding tokens excluded from loss so they don't corrupt gradients

---

## Step 1 — Format Using a Prompt Template

Each raw (instruction, response) pair is wrapped in a template. The most common is the Alpaca format:

```
### Instruction:
Convert 45 kilometers to meters.

### Response:
45 kilometers is 45,000 meters.
```

This adds delimiters that the model learns to recognize as task boundaries. At inference time, you provide everything up to and including `### Response:` and the model continues from there.

---

## Step 2 — Tokenize

The formatted string is converted to integer token IDs using the BPE tokenizer:

```
"### Instruction: Convert..." → [21106, 318, 281, 12064, 326, ...]
```

Each word or sub-word maps to one number in the 50,257-word vocabulary.

---

## Step 3 — Pad to Batch Length

Sequences in a batch are different lengths. They must be padded to the same length so PyTorch can stack them into a single tensor.

Padding uses the `<|endoftext|>` token (ID 50256) appended at the end:

```
Short sequence:  [21106, 318, 281, 13]
After padding:   [21106, 318, 281, 13, 50256, 50256, 50256]
                                       ↑ padding tokens
```

Unlike classification fine-tuning (which pads to a global max across the entire dataset), instruction fine-tuning pads **per batch** — only to the longest sequence in the current batch. This saves memory since instruction lengths vary enormously.

---

## Step 4 — Create the Target (Shift Left by 1)

Instruction fine-tuning is still next-token prediction. The target for each input position is the token that comes next. So the target sequence is the input shifted one position to the left, with one extra `50256` appended at the tail to keep the same length:

```
Input:   [21106,   318,   281, 12064, ..., 50256, 50256, 50256]
Target:  [  318,   281, 12064,   326, ..., 50256, 50256, 50256, 50256]
          ↑ shifted left by 1                              ↑ extra token at tail
```

Both input and target always have identical length. This is the same next-token prediction setup used during pretraining — the only difference is the data now consists of instruction-response pairs.

---

## Step 5 — Replace Padding Tokens in Target with -100

Not all `50256` tokens in the target are equal:
- The **first `50256`** is a real end-of-text token — the model should learn to predict it
- All subsequent `50256` tokens are pure padding — the model should not be penalized for them

```
Target before: [318, 281, ..., 13,  50256, 50256, 50256, 50256]
                                    ↑ real EOS    ↑ padding
Target after:  [318, 281, ..., 13,  50256,  -100,  -100,  -100]
```

PyTorch's `cross_entropy` automatically skips any position with target `-100` (this is the built-in `ignore_index`). Those positions contribute zero loss and zero gradient — the model is not updated based on padding tokens.

---

## Why -100 Specifically?

`-100` is PyTorch's default `ignore_index` for `nn.functional.cross_entropy`. It is a convention built into PyTorch — not a magic number you pick. Those positions are completely excluded from loss computation, not just zeroed.

---

## Full Pipeline at a Glance

```
Raw data:
  instruction: "Convert 45 km to meters."
  response:    "45 km is 45,000 meters."
         ↓
Step 1: Format
  "### Instruction: Convert 45 km...\n### Response: 45 km is 45000 meters."
         ↓
Step 2: Tokenize
  [21106, 318, 281, 12064, ..., 2231, 3967, 13]
         ↓
Step 3: Pad to batch max length
  [21106, 318, ..., 13, 50256, 50256, 50256]
         ↓
Step 4: Shift to create targets
  inputs:  [21106, 318, ..., 13,    50256, 50256, 50256]
  targets: [  318, 281, ..., 50256, 50256, 50256, 50256]
         ↓
Step 5: Mask padding in targets
  targets: [  318, 281, ..., 50256,  -100,  -100,  -100]
```

---

## Comparison with Classification Fine-Tuning

| | Classification FT | Instruction FT |
|---|---|---|
| Padding length | Global max across entire dataset | Per-batch max |
| Target | Single integer label | Token IDs shifted left by 1 |
| Loss masking | Not needed | Padding tokens → -100 |
| End token | pad_token_id=50256 throughout | First 50256 is real EOS; rest → -100 |

---

## Related

- [[fine-tuning]]
- [[instruction-finetuning-training-mechanics]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[tokenization]]
- [[instruction-finetuning-collate-padding-trick]]
