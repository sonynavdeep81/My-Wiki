---
title: Instruction Fine-Tuning
type: concept
tags: [fine-tuning, instruction-tuning, loss-masking, padding, collate, alpaca]
sources: 2
updated: 2026-05-05
verified_against: classification_fine_tuning, 2026-05-05
confidence: high
---

## Instruction Fine-Tuning

**Summary**: Fine-tune a pre-trained LLM to follow natural language instructions by training on (instruction, response) pairs with loss computed only on the response tokens.

## Key Distinction from Classification Fine-Tuning

| | Classification FT | Instruction FT |
|---|---|---|
| Output head | Replaced (Linear → num_classes) | Kept (original LM head) |
| Labels | Single integer per sample | Token IDs shifted left by 1 |
| Padding | Global max across dataset | Per-batch max (dynamic) |
| Loss masking | None needed | Instruction tokens + padding → -100 |
| Dataset size | Small (100s–1000s) | Large (10K–100K+) |

## Data Format

Raw entry (JSON):
```json
{"instruction": "...", "input": "...", "output": "..."}
```

Wrapped via prompt template (e.g. Alpaca):
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Full text = instruction_plus_input + `\n\n### Response:\n` + output.

## 5-Step Data Pipeline

1. **Format** — wrap (instruction, input, output) in chosen template
2. **Tokenize** — encode full text; truncate at `context_length` (truncation risk: may cut response)
3. **Pad** — within each batch, pad to `max_len+1` with `50256` (`<|endoftext|>`)
4. **Shift target** — `inputs = padded[:-1]`, `targets = padded[1:]`; extra +1 token ensures last target position exists
5. **Mask** — replace all padding `50256`s in targets (beyond first) with `-100`; first `50256` kept as real EOS prediction target

## Loss Masking

- `-100` = PyTorch `ignore_index` for `cross_entropy` — positions excluded from loss entirely
- Only response tokens contribute to weight updates
- Instruction tokens in target also set to `-100` (model reads them, doesn't get graded on them)

```python
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index  # keep first 50256, mask the rest
```

## collate_fn / Dynamic Padding

Padding to global max wastes compute. `custom_collate` pads each batch to its own `max_len`:
- Sequences within a batch padded to `max(len(item) for item in batch) + 1`
- Passed to `DataLoader` via `collate_fn=custom_collate`
- All sequences in batch become equal length → stackable tensor

## Truncation Risk

If tokenized full text > `context_length`, slicing `[:context_length]` may cut into the response. Fix: filter out samples exceeding context_length before training.

## Related

- [[fine-tuning]]
- [[stanford-alpaca]]
- [[instruction-finetuning-data-pipeline]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[instruction-finetuning-training-mechanics]]
- [[instruction-finetuning-collate-padding-trick]]
- [[dropout-during-finetuning]]
- [[lora]]
