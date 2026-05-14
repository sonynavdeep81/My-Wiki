---
title: Instruction Fine-Tuning — Collate Padding Trick (batch_max_length +1)
type: query
tags: [fine-tuning, instruction-tuning, padding, collate, dataloader, target-shift]
sources: 1
updated: 2026-05-14
---

## Instruction Fine-Tuning — Collate Padding Trick (batch_max_length +1)

**Summary**: The collate function pads sequences to `max_len+1` so that inputs and targets of equal length can be cleanly sliced from the same padded sequence without any extra bookkeeping.

---

## The Problem It Solves

In next-token prediction, for every input sequence of length N, you need a target sequence of the same length N — shifted one position to the right. If you pad first and shift second, you end up one token short on the target side. The `+1` trick solves this elegantly.

---

## The Code

```python
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)  # +1 extra slot

    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]                           # append one EOS/pad token
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])                   # drop last → inputs
        inputs_lst.append(inputs)
```

---

## Step-by-Step Walkthrough

Let's trace through one example: an item `[A, B, C, D]` of length 4, and `batch_max_length = 7`.

**Step 1 — Append one EOS token:**
```
[A, B, C, D]  →  [A, B, C, D, 50256]   length = 5
```

**Step 2 — Pad to batch_max_length (7):**
```
[A, B, C, D, 50256, 50256, 50256]       length = 7
```

**Step 3 — Slice to get inputs and targets:**
```
padded   = [A, B, C, D, 50256, 50256, 50256]   length = 7  (max_len + 1)

inputs   = padded[:-1]                          length = 6  (max_len)
         = [A, B, C, D, 50256, 50256]

targets  = padded[1:]                           length = 6  (max_len)
         = [B, C, D, 50256, 50256, 50256]
```

Both `inputs` and `targets` are exactly length 6. Same length, naturally aligned, no extra code needed.

---

## Why Add +1 Then Remove It?

The `+1` creates one extra slot at the end of the padded sequence. This slot serves as the **final target token** after the `[1:]` shift. Without it:

```
# Without +1:
padded   = [A, B, C, D, 50256, 50256]   length = 6
inputs   = padded[:-1] = [A, B, C, D, 50256]   length = 5
targets  = padded[1:]  = [B, C, D, 50256, 50256]   length = 5  ✓ same length

# But now the final target for the last padding token is missing.
# The +1 ensures there's always a valid target for every input position.
```

The extra `50256` at the tail of `padded` becomes the **last element of targets** after `[1:]` slicing — it provides a target for the final input position. This is not wasted; it is intentional.

---

## What Happens to Padding Tokens in the Target?

After creating targets, padding tokens are replaced with `-100`:

```
targets before: [B, C, D, 50256, 50256, 50256]
targets after:  [B, C, D, 50256,   -100,  -100]
```

- The **first `50256`** (real end-of-text token) stays — the model should learn to predict end-of-sequence
- Pure padding `50256`s beyond the first become `-100`
- PyTorch's `cross_entropy` skips any position with target `-100` — no loss, no gradient, no weight update on those positions

---

## Why -100 Specifically?

`-100` is PyTorch's built-in `ignore_index` for `nn.functional.cross_entropy`. It is not just zeroed — those positions are completely excluded from the loss computation. This is a PyTorch convention, not a magic number you can change without specifying `ignore_index` explicitly.

---

## Visual Summary

```
Original:     [A,  B,  C,  D]                          length = 4
+ EOS:        [A,  B,  C,  D,  50256]                  length = 5
+ padding:    [A,  B,  C,  D,  50256, 50256, 50256]    length = 7 (max+1)

inputs [:-1]: [A,  B,  C,  D,  50256, 50256]           length = 6
targets [1:]: [B,  C,  D,  50256, 50256, 50256]        length = 6

After -100:   [B,  C,  D,  50256,  -100,  -100]        ← loss computed only here ✓
```

---

## Related

- [[instruction-finetuning-data-pipeline]]
- [[instruction-finetuning-training-mechanics]]
- [[fine-tuning]]
- [[dataloader-parameters]]
