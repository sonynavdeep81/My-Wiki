---
title: Positional Embeddings
type: concept
tags: [positional-embeddings, transformer, rope, sinusoidal, learnable]
sources: 2
updated: 2026-04-14
---

## Positional Embeddings

**Summary**: Vectors added to token embeddings to inject position information, solving the transformer's permutation-invariance problem; three main variants exist — sinusoidal (original), learnable (GPT-2), and RoPE (modern).

## The Problem

The [[transformer-architecture]] is **permutation invariant** — self-attention computes the same result regardless of token order. Without positional information, "The cat bit the dog" and "The dog bit the cat" are indistinguishable (same token IDs, different order).

Positional embeddings inject a **unique signal per position** so the model can differentiate them.

## Three Variants

### 1. Sinusoidal (Original Transformer)

Uses fixed sine/cosine functions at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- No learned parameters — fully deterministic
- Can extrapolate to sequence lengths unseen during training
- Used in the original "Attention is All You Need" paper

### 2. Learnable / Absolute (GPT-2)

Position indices [0, 1, 2, ...] are passed through a second `nn.Embedding` layer:

```python
pos_emb = nn.Embedding(num_embeddings=context_size, embedding_dim=768)
position_vectors = pos_emb(torch.arange(T))  # shape: (T, 768)
final_input = token_vectors + position_vectors
```

- Parameters learned via backpropagation
- `num_embeddings` = context size (max sequence length the model can handle)
- Used by GPT-2; simple but cannot extrapolate beyond trained context length
- See [[embeddings]] for the full input construction

### 3. Rotary Positional Embeddings (RoPE)

Encodes position by **rotating** Q and K vectors in the attention computation rather than adding to embeddings:

```
Q_rotated = rotate(Q, position)
K_rotated = rotate(K, position)
```

The dot product Q·Kᵀ then naturally encodes **relative** position between tokens, not absolute.

- No extra parameters added to embeddings
- Relative position awareness — "token i is 3 positions before token j"
- Extrapolates better to longer contexts than learnable
- Used by **LLaMA**, Mistral, Falcon, and most modern open-source LLMs
- *Not yet covered in detail in this wiki — see LLaMA 2 paper*

## Comparison

| Variant | Learned | Type | Extrapolates | Used By |
|---|---|---|---|---|
| Sinusoidal | No | Absolute | Yes (limited) | Original Transformer |
| Learnable | Yes | Absolute | No | GPT-2 |
| RoPE | No | Relative | Better | LLaMA, Mistral, Falcon |

## Related

- [[embeddings]]
- [[transformer-architecture]]
- [[multi-head-attention]]
- [[decoder-only-architecture]]
- [[llama]]
