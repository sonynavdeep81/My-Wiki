---
title: SpamDataset — Classification Fine-Tuning Dataset Implementation
type: query
tags: [fine-tuning, classification, pytorch, dataset, tokenization, padding, truncation]
sources: 0
updated: 2026-05-14
---

## SpamDataset — Classification Fine-Tuning Dataset Implementation

**Summary**: A PyTorch `Dataset` class for SMS spam classification that handles tokenization, truncation to a maximum length, padding to a uniform length, and label conversion.

---

## The Full Implementation

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_tokens=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Tokenize all texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]

        # Determine max sequence length
        if max_tokens is None:
            self.max_tokens = max(len(e) for e in self.encoded_texts)
        else:
            self.max_tokens = max_tokens

        # Truncate sequences that are too long
        self.encoded_texts = [e[:self.max_tokens] for e in self.encoded_texts]

        # Pad sequences that are too short
        self.encoded_texts = [
            e + [pad_token_id] * (self.max_tokens - len(e))
            for e in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded_text = self.encoded_texts[idx]
        label = self.data.iloc[idx]['label']
        return torch.tensor(encoded_text), torch.tensor(label, dtype=torch.long)
```

---

## What Each Step Does

### Step 1 — Tokenize

```python
self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]
```

Each SMS text is converted to a list of integer token IDs using tiktoken's BPE tokenizer. Different texts produce lists of different lengths — a short "Win a prize!" becomes ~5 tokens; a long message might become 100+ tokens.

### Step 2 — Determine max_tokens

```python
if max_tokens is None:
    self.max_tokens = max(len(e) for e in self.encoded_texts)
else:
    self.max_tokens = max_tokens
```

`max_tokens` controls the sequence length. If not specified, it is set to the length of the longest sequence in this split. If specified manually, it overrides this automatic calculation.

**Important constraint:** `max_tokens` must never exceed `context_length` (256 for our GPT-2). The positional embedding table only has rows for positions 0 to 255. Exceeding this crashes the model.

### Step 3 — Truncate

```python
self.encoded_texts = [e[:self.max_tokens] for e in self.encoded_texts]
```

Any sequence longer than `max_tokens` is cut off. Only the first `max_tokens` tokens are kept. This ensures no sequence ever asks for a positional embedding row that doesn't exist.

### Step 4 — Pad

```python
self.encoded_texts = [
    e + [pad_token_id] * (self.max_tokens - len(e))
    for e in self.encoded_texts
]
```

Any sequence shorter than `max_tokens` has padding tokens appended until it reaches `max_tokens`. `pad_token_id=50256` is GPT-2's `<|endoftext|>` token repurposed as a padding sentinel.

After truncation and padding, **every sequence is exactly `max_tokens` tokens long**. PyTorch requires this uniformity to stack sequences into a batch tensor.

### Step 5 — Return Tensors

```python
return torch.tensor(encoded_text), torch.tensor(label, dtype=torch.long)
```

`dtype=torch.long` (int64) is required for `CrossEntropyLoss`. If you use the default float dtype for labels, PyTorch raises a `RuntimeError`.

---

## Why Truncate Before Padding?

The order matters:
1. **Truncate first** — reduces any long sequence to exactly `max_tokens` tokens
2. **Pad second** — fills any short sequence up to `max_tokens`

If you padded first and then truncated, you might accidentally cut off real content.

---

## Using the Same max_tokens Across All Splits

When creating val and test datasets, always pass `max_tokens` from the training dataset:

```python
train_dataset = SpamDataset("train.csv", tokenizer)          # max_tokens auto-computed
val_dataset   = SpamDataset("val.csv",   tokenizer, max_tokens=train_dataset.max_tokens)
test_dataset  = SpamDataset("test.csv",  tokenizer, max_tokens=train_dataset.max_tokens)
```

**Why:** The model was trained on sequences of a specific length. If val sequences are a different length, the batch shapes are inconsistent. Using the same `max_tokens` everywhere ensures all splits have identical sequence lengths.

---

## Why max_tokens Is a Parameter (Not Hardcoded)

Two legitimate use cases:

**1. Consistency across splits** (shown above) — pass training max_tokens to val/test.

**2. Intentional speed/memory trade-off** — truncate to a smaller value to speed up training:

```python
train_dataset = SpamDataset("train.csv", tokenizer, max_tokens=100)
# Only first 100 tokens of each message — faster, less memory
```

For spam detection, most of the signal is in the first 50–100 tokens anyway.

---

## Key Design Decisions Summary

| Decision | Reason |
|---|---|
| Truncate before pad | Ensures truncation cuts real content, not padding |
| `dtype=torch.long` for labels | Required by `CrossEntropyLoss` |
| `max_tokens` from train data | Ensures consistent sequence length across all splits |
| `pad_token_id=50256` | GPT-2's `<\|endoftext\|>` repurposed as padding |
| `max_tokens ≤ context_length` | Positional embedding table has fixed size; exceeding it crashes the model |

---

## Related

- [[fine-tuning]]
- [[context-length-assert]]
- [[dataloader-parameters]]
- [[gpt2-from-scratch]]
