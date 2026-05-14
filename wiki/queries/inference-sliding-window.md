---
title: Inference Sliding Window — Handling Context Length During Generation
type: query
tags: [inference, context-length, positional-embeddings, generation, sliding-window]
sources: 1
updated: 2026-05-14
---

## Inference Sliding Window — Handling Context Length During Generation

**Summary**: During inference, the generated sequence grows token by token and can exceed `context_length`. Since the positional embedding table has a fixed number of rows, the model must only ever see the last `context_length` tokens at each step.

---

## Why Training Has No Problem

During training, every sample is pre-built by the sliding window in `GPTDataset` to be exactly `context_length` tokens long. The input shape is always `(batch_size, context_length)`. The positional embedding lookup always asks for rows 0 through 255 — all valid.

---

## Why Inference Creates a Problem

During inference, the model generates one token at a time and appends it to the growing sequence:

```
Step 1:   "Every"                          → 1 token  → pos_emb[0]          ✓
Step 2:   "Every effort"                   → 2 tokens → pos_emb[0:1]        ✓
Step 3:   "Every effort moves"             → 3 tokens → pos_emb[0:2]        ✓
...
Step 256: "Every effort moves you ..."     → 256 tokens → pos_emb[0:255]    ✓
Step 257: "Every effort moves you ..."     → 257 tokens → pos_emb[256]      ✗ CRASH
```

The positional embedding table is `context_length × 768` — it only has rows 0 to 255. Asking for row 256 causes an index out-of-bounds error and crashes the program.

---

## The Fix — Sliding Window

Before passing the token sequence to the model, always take only the last `context_length` tokens:

```python
for i in range(max_length):
    token_ids_cond = token_ids[:, -context_size:]  # keep last 256 tokens only
    logits = model(token_ids_cond)
    ...
```

Now at every step, the model sees at most 256 tokens — always valid for the positional embedding table:

```
Step 257: 257 tokens → take last 256 → pos_emb[0:255] ✓
Step 258: 258 tokens → take last 256 → pos_emb[0:255] ✓
Step N:   N tokens   → take last 256 → pos_emb[0:255] ✓
```

The model loses access to the earliest tokens once the sequence grows beyond `context_size`, but it can generate indefinitely without crashing.

---

## The Trade-Off

When the sequence exceeds `context_length`, the model forgets the beginning. In the example above, after 257 tokens, "Every" (the first word) is no longer visible to the model. The model only ever has a window of the most recent 256 tokens.

This is an inherent limitation of fixed-context models. KV-caching and longer context models (e.g., with 8K or 128K context) are approaches that push this limit further, but the fundamental constraint remains.

---

## The Bug in gpt2_decoder.py

The original script defines `context_size` but forgets to apply the slice:

```python
# Buggy version:
context_size = GPT_CONFIG_124M["context_length"]   # defined...
logits = model(token_ids)                           # ...but never used here

# Fixed version:
logits = model(token_ids[:, -context_size:])        # slice to last context_size tokens
```

Without the slice, the model crashes as soon as the generated sequence exceeds 256 tokens.

---

## Related

- [[context-length-assert]]
- [[positional-embeddings]]
- [[decoding-strategies]]
- [[kv-caching]]
- [[input-to-output-workflow]]
