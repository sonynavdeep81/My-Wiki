---
title: SpamDataset — Truncation and Padding Lines Explained
type: query
tags: [fine-tuning, classification, spam, dataset, padding, truncation, sequence-length, pytorch]
sources: 1
updated: 2026-05-14
---

## SpamDataset — Truncation and Padding Lines Explained

**Summary**: Two consecutive lines in SpamDataset handle truncation and padding to ensure every sequence is exactly `max_tokens` long. The model itself accepts any sequence length up to 1024, but PyTorch requires all sequences in a batch to be the same length.

---

## The Two Lines

```python
# Line 1 — Truncate: cut any email longer than max_tokens
self.encoded_texts = [encoded_text[:self.max_tokens] for encoded_text in self.encoded_texts]

# Line 2 — Pad: extend any email shorter than max_tokens with pad token 50256
self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_tokens - len(encoded_text)) for encoded_text in self.encoded_texts]
```

Both lines always execute — they are not inside any `if/else`. But for any given email, **only one of them does anything meaningful**.

---

## What Each Line Does Per Email

| Email length | Line 1 (truncate) | Line 2 (pad) | Final length |
|---|---|---|---|
| Longer than `max_tokens` | Cuts to `max_tokens` | `max_tokens - max_tokens = 0` → nothing added | `max_tokens` |
| Shorter than `max_tokens` | Nothing cut | Appends pad tokens to reach `max_tokens` | `max_tokens` |
| Exactly `max_tokens` | Nothing cut | Nothing added | `max_tokens` |

After both lines, every email is **exactly `max_tokens` long**.

---

## Why Line 1 Must Come Before Line 2

Padding is always added at the end. If Line 2 ran first on a long email, `max_tokens - len(email)` would be negative, Python would add zero pad tokens, and the email would stay longer than `max_tokens`. Line 1 must run first to cut it down, so Line 2 has nothing to do.

---

## Why Line 1 Is Needed at All

`max_tokens` is computed from the **training set** — it equals the length of the longest email in the training CSV. There is no guarantee that val or test emails are all shorter than this. A val or test email could be longer than any email in the training set.

Without Line 1, that longer email would not be truncated, its length would differ from all other sequences, and PyTorch would crash when trying to stack the batch.

---

## Why the Crash Happens in the DataLoader, Not the Model

The model itself does not crash on variable-length sequences. GPT-2 can process any sequence up to 1024 tokens regardless of what length it was trained on.

The crash happens **before the model sees any data** — in the DataLoader, when PyTorch tries to stack individual samples into a batch tensor:

```
[120 tokens]
[150 tokens]   ← different length — cannot stack into a single tensor
[120 tokens]
```

PyTorch requires all sequences in a batch to be the same length. If even one sequence differs, stacking fails with a shape error.

---

## The Model Accepts Any Sequence Length — But Classification Quality Breaks

Once all sequences in a batch are the same length, the model processes them without crashing — even if that length differs from what it saw during training. The only hard limit is 1024, the size of the positional embedding table.

However, for classification the results will be poor if the sequence length differs from training.

**Example:** model trained on length 120, validation batches of length 150 — all same length, so PyTorch stacks fine, no crash.

But during training, `[:, -1, :]` always read from **position 119**. The last transformer block, final norm, and out_head were all trained to produce the spam/ham decision from whatever sits at position 119. During validation with length 150, `[:, -1, :]` reads from **position 149** — a position the trainable layers were never updated for. The classification output will be unreliable.

The reason `max_tokens` is kept consistent across train, val, and test is not a model constraint — it is to ensure `[:, -1, :]` always reads from the same position, so the trainable layers see the same kind of input they were trained on.

---

## Related

- [[classification-finetuning-workflow]]
- [[padding-strategy-classification-vs-instruction]]
- [[spam-dataset-implementation]]
- [[fine-tuning]]
