---
title: Bias Comparison — GPT-2 vs Attention Is All You Need
type: query
tags: [bias, gpt2, attention-is-all-you-need, architecture, comparison]
sources: 2
updated: 2026-04-16
---

## Bias Comparison — GPT-2 vs Attention Is All You Need

**Summary**: Which linear layers use bias in GPT-2 (scratch vs OpenAI checkpoint) vs the original Transformer paper.

| Component | GPT-2 (train from scratch) | GPT-2 (OpenAI checkpoint) | Attention Is All You Need |
|---|:---:|:---:|:---:|
| Token Embedding | No (lookup, no bias) | No | No (lookup, no bias) |
| Positional Embedding | No (lookup, no bias) | No | N/A — sinusoidal, no params |
| Q, K, V projections | **No** (`qkv_bias=False`) | **Yes** (`qkv_bias=True`) | **Yes** |
| Output projection W_O | Yes | Yes | Yes |
| FFN Layer 1 | Yes | Yes | Yes (explicit `b₁` in formula) |
| FFN Layer 2 | Yes | Yes | Yes (explicit `b₂` in formula) |
| LayerNorm β | Yes (learned shift) | Yes | Yes |
| Output head (`lm_head`) | **No** (`bias=False`) | **No** (`bias=False`) | **No** (weight-tied, no bias) |

## Key Notes

**Q/K/V bias — the main divergence:**
- The [[attention-is-all-you-need]] paper uses standard linear projections with bias.
- GPT-2 trained from scratch drops Q/K/V bias because [[layer-normalization]]'s β parameter already provides the learned offset — the extra bias is redundant.
- The OpenAI-released checkpoint *does* include Q/K/V bias tensors. This is why the `qkv_bias` flag exists in `MultiHeadAttention.__init__`: it must be `True` to load OpenAI weights (shape mismatch otherwise).

**Positional encoding:**
- The paper uses fixed sinusoids — no parameters, no bias, no trainable weights at all.
- GPT-2 uses `nn.Embedding(context_length, emb_dim)` — still no bias, but the table itself is learned.

**Output head (`lm_head`) — both agree: no bias:**
- The output projection is [[weight-tying|weight-tied]] to `tok_emb` (same tensor).
- Biases are not part of the weight-tying relationship, so both implementations omit them for consistency.

## Related

- [[gpt2-from-scratch]]
- [[attention-is-all-you-need]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[layer-normalization]]
- [[weight-tying]]
- [[positional-embeddings]]
