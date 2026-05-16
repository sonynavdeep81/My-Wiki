---
title: Padding Strategy — Classification vs Instruction Fine-Tuning
type: query
tags: [fine-tuning, classification, instruction-tuning, padding, dataset, last-token, dynamic-padding]
sources: 2
updated: 2026-05-14
---

## Padding Strategy — Classification vs Instruction Fine-Tuning

**Summary**: Classification fine-tuning pads all sequences to the training-set maximum length; instruction fine-tuning pads per batch. The reason is that classification reads only the last token's output, so the last position must be consistent across all batches.

---

## Classification Fine-Tuning — Why Static (Whole-Dataset) Padding

The classification decision comes from one position only:

```python
logits = model(batch)[:, -1, :]   # last token's output → shape (batch_size, 2)
loss = nn.functional.cross_entropy(logits, targets)
```

The trainable components are: `out_head`, `trf_blocks[-1]`, and `final_norm`. All three receive their input from whichever token sits at the last position.

**With dynamic padding (per-batch) — the problem:**

Take the email "Win a free iPhone" (4 tokens).

- Batch 1, max length 10 → last position is 9 → `trf_blocks[-1]`, `final_norm`, and `out_head` are updated based on **PAD token at position 9**
- Batch 2, max length 20 → last position is 19 → the same three components are updated based on **PAD token at position 19**

The same email produces updates to the trainable layers from two different inputs (PAD@9 vs PAD@19) in different batches. The trainable layers never get a consistent signal and cannot learn reliably.

**With static padding (whole-dataset max) — the fix:**

```python
max_tokens = max(len(enc) for enc in train_encoded_texts)
```

Every sequence is always padded to the same length. `[:, -1, :]` always picks the same position in every batch. `trf_blocks[-1]`, `final_norm`, and `out_head` are always updated based on the PAD token at that one fixed position — consistent training signal every time.

The same `max_tokens` is passed explicitly to val and test datasets:

```python
train_dataset = SpamDataset("train.csv", tokenizer)
val_dataset   = SpamDataset("val.csv",   tokenizer, max_tokens=train_dataset.max_tokens)
test_dataset  = SpamDataset("test.csv",  tokenizer, max_tokens=train_dataset.max_tokens)
```

---

## Instruction Fine-Tuning — Why Per-Batch (Dynamic) Padding Works

The loss is computed at every token position:

```python
loss = cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
```

PAD positions are masked with `ignore_index=-100` and contribute zero loss. Every real token position trains independently on its own content — predict the next token. There is no single "answer position." Varying batch lengths just change how many positions get trained in a given batch, not what job any position does.

---

## Why the Last Token Is Used for Classification

Due to causal masking, each token attends only to itself and all tokens before it. The last token has therefore attended to the entire input sequence and its output is the richest summary of the full email — the most informed representation for making a spam/ham decision.

---

## Can Classification Use Per-Batch Padding?

Yes, but `[:, -1, :]` must be replaced with per-example indexing of the last real token:

```python
last_token_idx = sequence_lengths - 1
logits = model(batch)[range(batch_size), last_token_idx, :]
```

This picks the last real token regardless of padding length, giving consistent inputs to the trainable layers. It requires tracking per-example lengths through the DataLoader — more complex, and unnecessary for short SMS texts.

---

## Summary

| | Instruction Fine-Tuning | Classification Fine-Tuning |
|---|---|---|
| Output used | All positions | Last position only |
| Padding scope | Per batch | Whole training set |
| Why | Each position trains independently | Trainable layers must see the same input position every batch |
| PAD token | 50256 | 50256 |
| PAD masked? | Yes — `ignore_index=-100` | Not needed |

---

## Related

- [[fine-tuning]]
- [[classification-finetuning-workflow]]
- [[spam-dataset-implementation]]
- [[instruction-finetuning-collate-padding-trick]]
- [[cross-entropy-loss]]
