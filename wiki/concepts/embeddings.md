---
title: Embeddings (Token & Positional)
type: concept
tags: [embeddings, positional-encoding, representation, transformer]
sources: 1
updated: 2026-04-13
---

## Embeddings (Token & Positional)

**Summary**: Dense vector representations that convert discrete token IDs and position indices into continuous 768-dimensional vectors that the transformer can process.

## Token Embeddings

Token IDs (integers) carry no semantic information on their own. An **embedding lookup table** maps each token ID to a learned 768-dimensional vector.

- GPT-2 table shape: **50,257 × 768**
- Rows = one per vocabulary token (indexed 0–50,256)
- Columns = 768 dimensions of the embedding space
- Vectors are randomly initialized, then **learned via backpropagation** during training

```python
embedding = nn.Embedding(num_embeddings=50257, embedding_dim=768)
token_vectors = embedding(token_ids)  # shape: (T, 768)
```

## Positional Embeddings

The [[transformer-architecture]] is **permutation invariant** — self-attention treats shuffled inputs identically. Without position information, "The cat bit the dog" and "The dog bit the cat" are indistinguishable.

Positional embeddings inject a unique signal per position:

```python
position_ids = torch.arange(len(token_ids))
pos_embedding = nn.Embedding(num_embeddings=256, embedding_dim=768)
position_vectors = pos_embedding(position_ids)  # shape: (T, 768)
```

- `num_embeddings` here = **context size** (max tokens the model can process at once)
- GPT used **learnable** positional embeddings (not sinusoidal formulas like the original Transformer paper)

## Combined Input

```
final_input_vectors = token_vectors + position_vectors  # shape: (T, 768)
```

This summed representation is the input to the first transformer block.

## Related

- [[tokenization]]
- [[byte-pair-encoding]]
- [[positional-embeddings]]
- [[transformer-architecture]]
- [[decoder-only-architecture]]
