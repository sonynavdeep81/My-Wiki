---
title: Byte-Pair Encoding (BPE)
type: concept
tags: [tokenization, bpe, vocabulary, nlp]
sources: 1
updated: 2026-04-13
---

## Byte-Pair Encoding (BPE)

**Summary**: An iterative algorithm that builds a subword vocabulary by repeatedly merging the most frequent adjacent token pair, used universally by modern LLMs for tokenization.

## Algorithm

1. Start with a character-level base vocabulary
2. Count every adjacent pair of tokens across the training corpus
3. Merge the most frequent pair into a new token; assign it a new ID
4. Repeat until vocabulary size target is reached

### Example (Training data: "low"×5, "lower"×2, "newest"×6, "widest"×3)

Base vocab: {l, o, w, e, r, n, s, t, i, d} → IDs 0–9

| Step | Most Frequent Pair | Merged Token | New ID |
|---|---|---|---|
| 1 | (e, s) → 9 | es | 10 |
| 2 | (es, t) → 9 | est | 11 |
| 3 | (l, o) → 7 | lo | 12 |
| 4 | (lo, w) → 7 | low | 13 |
| 5 | (n, e) → 6 | ne | 14 |
| ... | ... | ... | ... |
| 12 | (low, er) → 2 | lower | 21 |

Final vocab (size 22): {l, o, w, e, r, n, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest, er, lower}

## Vocabulary Size and Trade-offs

- **Larger vocab** (e.g., 50k): words map directly to single tokens; faster inference
- **Smaller vocab**: words represented as combinations of subwords; slower but no unknown tokens
- BPE **never produces unknown tokens** — any word can be decomposed to individual characters in the worst case
- Modern LLMs (GPT-4, LLaMA): vocab sizes between **50k and 100k**
- GPT-2: exactly **50,257 tokens**

## Related

- [[tokenization]]
- [[embeddings]]
- [[tiktoken]]
