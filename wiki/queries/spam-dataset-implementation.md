---
title: SpamDataset — Classification Fine-Tuning Dataset Implementation
type: query
tags: [fine-tuning, classification, pytorch, dataset, tokenization, padding, truncation]
sources: 0
updated: 2026-04-22
---

## SpamDataset — Classification Fine-Tuning Dataset Implementation

**Summary**: PyTorch Dataset class for SMS spam classification; handles tokenization, truncation, padding, and tensor conversion.

## Full Implementation

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_tokens=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]

        if max_tokens is None:
            self.max_tokens = max(len(e) for e in self.encoded_texts)
        else:
            self.max_tokens = max_tokens

        # truncate
        self.encoded_texts = [e[:self.max_tokens] for e in self.encoded_texts]
        # pad
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

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Truncate before pad | Truncation reduces length first; padding then fills to exact `max_tokens` |
| `dtype=torch.long` for label | `CrossEntropyLoss` requires int64; float labels cause `RuntimeError` |
| `max_tokens` from training data | Ensures all sequences same length as model was trained on |
| `pad_token_id=50256` | GPT-2's `<\|endoftext\|>` token used as padding sentinel |

## Truncation vs Padding

- **Truncate** (`e[:max_tokens]`): fires when sequence > max_tokens; during training only if max_tokens manually lowered; always possible at inference
- **Pad** (`+ [pad_id] * (max_tokens - len(e))`): fires when sequence < max_tokens; PyTorch slicing never auto-pads — must be explicit

## max_tokens Across Splits

```python
train_dataset = SpamDataset("train.csv", tokenizer)
val_dataset   = SpamDataset("val.csv",   tokenizer, max_tokens=train_dataset.max_tokens)
test_dataset  = SpamDataset("test.csv",  tokenizer, max_tokens=train_dataset.max_tokens)
```

If max_tokens computed from training data → no truncation during training by definition (every sample ≤ its own max).

## Why max_tokens Parameter Exists

Two use cases — both require a value ≤ `context_length`:

**1. Consistency across splits** — val/test must use same max_tokens as training:
```python
val_dataset = SpamDataset("val.csv", tokenizer, max_tokens=train_dataset.max_tokens)
```

**2. Intentional limiting** — pass a smaller value to speed up training or reduce memory:
```python
train_dataset = SpamDataset("train.csv", tokenizer, max_tokens=100)  # only first 100 tokens
```

`max_tokens` must always satisfy: `max_tokens ≤ context_length` — enforced by the assert before training. Passing `max_tokens=300` with `context_length=256` will create valid sequences but crash at the first forward pass when positional embedding lookup exceeds its table size. See [[context-length-assert]].

## Related

- [[fine-tuning]]
- [[lora]]
- [[gpt2-from-scratch]]
- [[context-length-assert]]
- [[dataloader-parameters]]
