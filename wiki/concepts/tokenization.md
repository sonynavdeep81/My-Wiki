---
title: Tokenization
type: concept
tags: [tokenization, bpe, nlp, preprocessing]
sources: 1
updated: 2026-04-13
---

## Tokenization

**Summary**: The process of breaking raw text into discrete tokens (numeric IDs) that a model can process, with subword tokenization (BPE) being the universal standard for LLMs.

## Three Levels

| Type | Vocab Size | OOV Handling | Pros | Cons |
|---|---|---|---|---|
| **Word-based** | ~170–200k | Poor (fails on unseen words) | Short sequences | Misses morphological similarities |
| **Character-based** | ~256 | Excellent (no OOV) | Tiny vocab | Long sequences; letters carry little meaning |
| **Subword (BPE)** | ~50k (moderate) | Effective; breaks into sub-units or chars | Best of both worlds | More complex to implement |

All major LLM providers use **subword tokenization via [[byte-pair-encoding]]**.

## GPT-2 Example

```python
import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')
token_ids = tokenizer.encode("The cat sat on",
                              allowed_special={'<|endoftext|>'})
# → [464, 3797, 3332, 319]
```

`<|endoftext|>` (token ID 50256) is a special vocabulary entry used during pretraining to separate documents in the training corpus. [[tiktoken]] raises an error by default if special tokens appear in text; `allowed_special` opts them in.

## After Tokenization: Embeddings

Token IDs are integer indices into an embedding lookup table. In GPT-2: 50,257 tokens × 768 dimensions. See [[embeddings]].

## Related

- [[byte-pair-encoding]]
- [[embeddings]]
- [[large-language-models]]
- [[tiktoken]]
