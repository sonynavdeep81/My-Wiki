---
title: Instruction Fine-Tuning — Collate Padding Trick (batch_max_length +1)
type: query
tags: [fine-tuning, instruction-tuning, padding, collate, dataloader, target-shift]
sources: 1
updated: 2026-05-01
---

## Instruction Fine-Tuning — Collate Padding Trick (batch_max_length +1)

**Summary**: The collate function pads to `max_len+1` so inputs and targets can be cleanly sliced from the same padded sequence — inputs use `[:-1]`, targets use `[1:]`.

## The Code Pattern

```python
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)  # +1 extra slot

    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]                         # append one EOS/pad
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )                                                  # pad to batch_max_length
        inputs = torch.tensor(padded[:-1])                 # drop last → inputs
        inputs_lst.append(inputs)
```

## Why Add +1 Then Remove It?

The trick exists to create `inputs` and `targets` of equal length from one padded sequence.

```
padded (length = max_len+1):
  [A, B, C, D, 50256, 50256, 50256]

inputs  = padded[:-1]  →  [A, B, C, D, 50256, 50256]      ← length max_len
targets = padded[1:]   →  [B, C, D, 50256, 50256, 50256]   ← length max_len (shift left by 1)
```

- `inputs[:-1]` drops the last token — back to `max_len`
- `targets[1:]` shifts left by 1 — the extra `+1` slot provides the final target token naturally
- Both tensors end up the same length with no extra work

## The Extra 50256 Is Not Wasted

The comment in code says: *"the extra padding token will be relevant in later codes"*

That extra `50256` at the tail of `padded` becomes the **last token of targets** after `[1:]` slicing — it's the target for the final input position. Without `+1`, targets would be one token short.

## Step-by-Step for One Example

```
Original item:        [A, B, C, D]          length = 4
After += [50256]:     [A, B, C, D, 50256]   length = 5
batch_max_length = 7  (longest+1 in batch)
After padding:        [A, B, C, D, 50256, 50256, 50256]   length = 7

inputs  = padded[:-1] = [A, B, C, D, 50256, 50256]        length = 6
targets = padded[1:]  = [B, C, D, 50256, 50256, 50256]    length = 6
```

Targets then have padding `50256`s replaced with `-100` to exclude from loss.

## Related

- [[instruction-finetuning-data-pipeline]]
- [[instruction-finetuning-training-mechanics]]
- [[fine-tuning]]
- [[dataloader-parameters]]
