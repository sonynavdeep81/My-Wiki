---
title: Context Length Assert — Why max_tokens Must Not Exceed context_length
type: query
tags: [context-length, positional-embeddings, assert, dataset, truncation]
sources: 1
updated: 2026-05-14
---

## Context Length Assert — Why max_tokens Must Not Exceed context_length

**Summary**: Sequences longer than `context_length` crash the model because positional embedding rows beyond that index don't exist. The assert catches this before training starts.

---

## The Assert

```python
assert train_dataset.max_tokens <= GPT_CONFIG_124M['context_length'], (
    f"Dataset length {train_dataset.max_tokens} exceeds model's context "
    f"length {GPT_CONFIG_124M['context_length']}. Reinitialize with "
    f"max_tokens={GPT_CONFIG_124M['context_length']} otherwise problem "
    f"with Positional Embeddings will occur"
)
```

This runs once before training begins. If `max_tokens > context_length`, the program crashes immediately with a clear error message — much better than failing mid-training with a cryptic index error.

---

## Why Sequences Cannot Exceed context_length

The positional embedding table has exactly `context_length` rows — one per position:

```python
self.pos_emb = nn.Embedding(context_length, emb_dim)
# → a table of shape (1024, 768) for context_length=1024
# → valid row indices: 0, 1, 2, ..., 1023
```

During the forward pass, the model looks up position embeddings for every token:

```python
pos_emb = self.pos_emb(torch.arange(num_tokens))
# → looks up rows 0, 1, 2, ..., num_tokens-1
```

If `num_tokens = 1100`, it tries to look up row 1024, 1025, ..., 1099. Those rows **do not exist** in the table. PyTorch raises an index out-of-bounds error and training crashes.

---

## A Concrete Example

Say `context_length = 1024` and you pass a sequence of 1100 tokens:

```
token 0    → pos_emb[0]    ✓
token 1    → pos_emb[1]    ✓
...
token 1023 → pos_emb[1023] ✓
token 1024 → pos_emb[1024] ✗  row doesn't exist → CRASH
```

---

## Why the Model Doesn't Auto-Truncate

You might expect the model to simply ignore tokens beyond position 1023. It doesn't. The model tries to look up a positional embedding for every token in the input. There is no automatic truncation — that must happen **explicitly at the dataset level** before data ever reaches the model.

---

## The Fix — Truncate in the Dataset

```python
self.encoded_texts = [e[:self.max_tokens] for e in self.encoded_texts]
```

With `max_tokens = context_length`, every sequence is trimmed to at most 1024 tokens. The positional embedding table is never asked for an out-of-range row. The assert always passes.

---

## Two Scenarios

| Scenario | What to do |
|---|---|
| `max_tokens=None` (auto from data) | Computed from longest sequence; set a truncation cap equal to context_length |
| `max_tokens` manually set | Must be ≤ context_length; assert catches violations immediately |

---

## Related

- [[positional-embeddings]]
- [[spam-dataset-implementation]]
- [[embeddings]]
- [[fine-tuning]]
- [[inference-sliding-window]]
