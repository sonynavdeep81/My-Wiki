---
title: Inference Sliding Window — Handling Context Length During Generation
type: query
tags: [inference, context-length, positional-embeddings, generation, sliding-window]
sources: 1 (raw/gpt2_decoder.py — bug in generate function; notebook superseded)
updated: 2026-04-23
---

## Inference Sliding Window — Handling Context Length During Generation

**Summary**: During inference, generated sequence grows unboundedly; positional embedding table is fixed at context_length rows — sliding window keeps last context_length tokens to prevent index out of bounds crash.

## Training vs Inference

**Training — no problem:**
- Every sample pre-built to exactly `context_length` tokens via sliding window in `GPTDataset`
- Input shape always `(batch_size, context_length)` → `pos_emb[0:255]` → always valid

**Inference — problem arises over time:**

```
Step 1:   4 tokens   → pos_emb[0:3]   ✓
Step 2:   5 tokens   → pos_emb[0:4]   ✓
...
Step 253: 257 tokens → pos_emb[256]   ✗ CRASH — row doesn't exist
```

pos_emb table is `context_length × 768` — rows 0 to 255 only.

## Fix — Sliding Window

```python
for i in range(max_length):
    token_ids_cond = token_ids[:, -context_size:]  # always keep last 256 tokens
    logits = model(token_ids_cond)
    ...
```

```
Step 253: 257 tokens → take last 256 → pos_emb[0:255] ✓
Step 254: 258 tokens → take last 256 → pos_emb[0:255] ✓
Step N:   N tokens   → take last 256 → pos_emb[0:255] ✓
```

Model loses the earliest tokens when sequence exceeds `context_size` but can generate indefinitely without crashing.

## Bug in gpt2_decoder.py

`context_size` defined but slicing never applied in generate function:

```python
# Current (buggy)
context_size = GPT_CONFIG_124M["context_length"]  # defined but unused
logits = model(token_ids)                          # full token_ids — no slicing

# Fix
logits = model(token_ids[:, -context_size:])       # slice to last context_size tokens
```

## Related

- [[context-length-assert]]
- [[positional-embeddings]]
- [[decoding-strategies]]
- [[kv-caching]]
- [[input-to-output-workflow]]
