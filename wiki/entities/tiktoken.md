---
title: tiktoken
type: entity
tags: [tool, tokenization, openai, bpe]
sources: 1
updated: 2026-04-13
---

## tiktoken

**Summary**: OpenAI's fast BPE tokenizer library used to tokenize text for GPT models.

- Implements [[byte-pair-encoding]] for GPT-2, GPT-3, GPT-4 tokenizers
- GPT-2 vocab: 50,257 tokens
- Raises an error by default if text contains special tokens like `<|endoftext|>`; use `allowed_special` parameter to opt in
- `<|endoftext|>` (token ID 50256) separates documents in the pretraining corpus

```python
import tiktoken
tokenizer = tiktoken.encoding_for_model('gpt2')
ids = tokenizer.encode("The cat sat on",
                        allowed_special={'<|endoftext|>'})
# → [464, 3797, 3332, 319]
```

## Related

- [[tokenization]]
- [[byte-pair-encoding]]
- [[gpt-family]]
