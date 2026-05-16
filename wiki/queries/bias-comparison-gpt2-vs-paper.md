---
title: Bias Comparison — GPT-2 vs Attention Is All You Need
type: query
tags: [bias, gpt2, attention-is-all-you-need, architecture, comparison]
sources: 2
updated: 2026-05-14
---

## Bias Comparison — GPT-2 vs Attention Is All You Need

**Summary**: Which linear layers use bias in GPT-2 (from scratch vs OpenAI checkpoint) versus the original Transformer paper.

---

## What Is a Bias in a Linear Layer?

Every `nn.Linear` layer computes:

```
output = input @ weight + bias
```

The `bias` is an optional learned vector added after the matrix multiply. Setting `bias=False` removes it entirely — the layer only does the matrix multiply.

---

## The Full Comparison Table

| Component             | GPT-2 (train from scratch) | GPT-2 (OpenAI checkpoint) |   Attention Is All You Need   |
| --------------------- | :------------------------: | :-----------------------: | :---------------------------: |
| Token Embedding       |    No (lookup, no bias)    |            No             |     No (lookup, no bias)      |
| Positional Embedding  |    No (lookup, no bias)    |            No             |  N/A — sinusoidal, no params  |
| Q, K, V projections   | **No** (`qkv_bias=False`)  | **Yes** (`qkv_bias=True`) |            **Yes**            |
| Output projection W_O |            Yes             |            Yes            |              Yes              |
| FFN Layer 1           |            Yes             |            Yes            |              Yes              |
| FFN Layer 2           |            Yes             |            Yes            |              Yes              |
| LayerNorm β (shift)   |    Yes (learned shift)     |            Yes            |              Yes              |
| Output head (lm_head) |   **No** (`bias=False`)    |   **No** (`bias=False`)   | **No** (weight-tied, no bias) |

---

## The Main Difference — Q, K, V Projections

This is where GPT-2 from scratch and the original paper diverge.

**Original Transformer paper (Attention Is All You Need):** Uses standard linear projections with bias for Q, K, V — this was the default in 2017.

**GPT-2 trained from scratch (`qkv_bias=False`):** Drops the Q/K/V bias. The reason is that [[layer-normalization]] already has a learned `shift` (β) parameter that provides a learned offset for each embedding dimension. Adding another bias in Q, K, V would be redundant — two parameters doing the same job.

**OpenAI's released GPT-2 checkpoint (`qkv_bias=True`):** The original OpenAI training _did_ include Q/K/V biases. This is purely historical — it was trained in 2019 before the community established that these biases are unnecessary. This is why the `qkv_bias` flag exists in `MultiHeadAttention.__init__`:

```python
self.W_query = nn.Linear(d_model, d_model, bias=self.qkv_bias)
```

If you set `qkv_bias=False` and then try to load OpenAI's pretrained weights, you get a shape mismatch crash — the saved weights include bias tensors but your layer has none.

---

## Positional Encoding — A Key Structural Difference

The original Transformer paper uses **fixed sinusoidal positional encoding** — a mathematical formula, no learned parameters, no bias, no weight matrix at all.

GPT-2 uses **learned positional embeddings** — an `nn.Embedding(context_length, emb_dim)` table that is trained from scratch. Like token embeddings, this is a lookup table (no bias — just a weight matrix).

---

## Output Head — Both Agree: No Bias

The output head (`lm_head`) maps the 768-dim hidden state back to 50,257 vocabulary scores. Both GPT-2 and the original paper use `bias=False` here.

The reason: the output head is [[weight-tying|weight-tied]] to the token embedding — they share the same weight matrix. Biases are not part of this sharing relationship, and adding a bias to the output head would create an asymmetry. Both implementations omit it for consistency and cleanliness.

---

## Summary in Plain Words

- **Embeddings:** Never have bias — they are lookup tables, not linear layers.
- **Q/K/V:** The paper uses bias; modern GPT-2 from scratch drops it (LayerNorm already provides the offset).
- **FFN and W_O:** Both use bias — standard practice.
- **Output head:** No bias in either — consistent with weight tying.
- **The `qkv_bias=True` flag** exists solely to load OpenAI's pretrained weights without crashing.

---

## Related

- [[gpt2-from-scratch]]
- [[attention-is-all-you-need]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[layer-normalization]]
- [[weight-tying]]
- [[positional-embeddings]]
