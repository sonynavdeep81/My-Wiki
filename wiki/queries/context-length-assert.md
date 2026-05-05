---
title: Context Length Assert — Why max_tokens Must Not Exceed context_length
type: query
tags: [context-length, positional-embeddings, assert, dataset, truncation]
sources: 1 (raw/classification_fine_tuning.py)
updated: 2026-04-24
---

## Context Length Assert — Why max_length Must Not Exceed context_length

**Summary**: Sequences longer than `context_length` crash the model because positional embedding rows beyond that index don't exist.

## The Assert

```python
assert train_dataset.max_tokens <= GPT_CONFIG_124M['context_length'], (
    f"Dataset length {train_dataset.max_tokens} exceeds model's context "
    f"length {GPT_CONFIG_124M['context_length']}. Reinitialize with "
    f"max_tokens={GPT_CONFIG_124M['context_length']} otherwise problem with Positional Embeddings will occur"
)
```

Safety check before training: if `max_tokens > context_length`, crash immediately with a helpful message rather than failing mid-training with a cryptic positional embedding index error.

## Rule

`max_tokens ≤ context_length` — always. Two scenarios:

| Scenario | What happens |
|---|---|
| `max_tokens=None` (auto from data) | Computed from longest sequence; truncate to context_length if needed |
| `max_tokens` manually set | User must pass value ≤ context_length; assert catches violations |

## Why It Crashes — Positional Embeddings

Positional embedding table has exactly `context_length` rows:

```
pos_emb table: context_length × 768   (e.g. 256 × 768)
Valid rows:    0, 1, 2, ... 255
```

Passing 300 tokens → model looks up positions 0–299:

```
token 0   → pos_emb[0]   ✓
...
token 255 → pos_emb[255] ✓
token 256 → pos_emb[256] ✗  index out of bounds — row doesn't exist → CRASH
```

## Why "Just Use Last 256 Tokens" Doesn't Work Automatically

The model does not auto-truncate. It tries to look up all positions in the input — including out-of-range ones. Truncation must happen **explicitly at the dataset level** before data ever reaches the model.

## The Fix — Truncate in SpamDataset

```python
self.encoded_texts = [e[:self.max_tokens] for e in self.encoded_texts]
```

With `max_tokens = context_length`, no sequence ever exceeds the positional embedding table size → assert always passes.

## Related

- [[positional-embeddings]]
- [[spam-dataset-implementation]]
- [[embeddings]]
- [[fine-tuning]]
- [[inference-sliding-window]]
