---
title: Instruction Fine-Tuning — Data Preparation Pipeline (5 Steps)
type: query
tags: [fine-tuning, instruction-tuning, padding, loss-masking, tokenization, target-shift]
sources: 1
updated: 2026-04-30
---

## Instruction Fine-Tuning — Data Preparation Pipeline (5 Steps)

**Summary**: Instruction fine-tuning data goes through 5 steps: format → tokenize → pad → create shifted target → mask padding tokens with -100.

## Step-by-Step Pipeline

### 2.1 Format using prompt template
Wrap raw (instruction, response) pair in the chosen template (e.g. Alpaca):
```
### Instruction: {instruction}
### Input: {input}
### Response: {response}
```

### 2.2 Tokenize
Convert formatted string to token IDs:
```
[21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, ...]
```

### 2.3 Pad to same length
Short sequences padded with `50256` (`<|endoftext|>`) at the end to match the longest sequence in the batch:
```
Input: [21106, 318, 281, ..., 13, 50256, 50256, 50256]
```

### 2.4 Create target (shift LEFT by 1)
Target = input shifted **left** by 1 position + one extra `50256` appended at the end:
```
Input:  [21106, 318, 281, 12064, ..., 50256, 50256, 50256]
Target: [  318, 281, 12064, 326, ..., 50256, 50256, 50256, 50256]
```
- Drop the first token from the front
- Append one `50256` at the tail to keep the same length
- Both input and target always have identical length

### 2.5 Replace padding tokens in target with -100
```
Target before: [318, 281, ..., 50256, 50256, 50256, 50256]
Target after:  [318, 281, ..., 50256, -100,  -100,  -100 ]
```
- The **first** `50256` (real end-of-text token) stays — model should learn to predict end-of-sequence
- Pure padding `50256`s become `-100`
- PyTorch `cross_entropy` skips any position labelled `-100` → no loss, no weight update on padding

## Why -100 Specifically?

`-100` is PyTorch's built-in `ignore_index` for `nn.functional.cross_entropy`. Any target position set to -100 is completely excluded from loss computation — not just zeroed, but skipped entirely.

## Key Distinction from Classification FT

| | Classification FT | Instruction FT |
|---|---|---|
| Padding length | Global max across entire dataset | Per-batch max |
| Target | Single integer label | Token IDs shifted left by 1 |
| Masking | No -100 masking needed | Padding tokens in target → -100 |
| End token | pad_token_id=50256 throughout | 50256 as real EOS + pad; only pad ones → -100 |

## Related

- [[fine-tuning]]
- [[instruction-finetuning-training-mechanics]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[stanford-alpaca]]
- [[tokenization]]
- [[instruction-finetuning-collate-padding-trick]]
