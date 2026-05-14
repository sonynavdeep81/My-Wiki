---
title: GPT-2 Implementation vs Attention Is All You Need — Parameter Comparison
type: query
tags: [comparison, gpt2, attention-is-all-you-need, hyperparameters]
updated: 2026-05-14
---

## GPT-2 Implementation vs Attention Is All You Need — Parameter Comparison

**Summary**: Side-by-side comparison of architecture and training hyperparameters between the GPT-2 decoder-only implementation and the original Transformer paper (Vaswani et al., 2017).

---

## The Full Comparison Table

| Parameter | GPT-2 Implementation | Attention Is All You Need (base) |
|---|---|---|
| **Architecture** | Decoder-only | Encoder + Decoder |
| **d_model** | 768 | 512 |
| **d_ff (FFN width)** | 3,072 (4×) | 2,048 (4×) |
| **Attention heads** | 12 | 8 |
| **d_k = d_v (head dim)** | 64 | 64 |
| **Layers** | 12 decoder layers | 6 encoder + 6 decoder |
| **Activation (FFN)** | GELU | ReLU |
| **Layer Norm placement** | Pre-LN (before attention) | Post-LN (after attention) |
| **Positional encoding** | Learnable (`nn.Embedding`) | Sinusoidal (fixed, no params) |
| **Dropout** | 0.1 | 0.1 |
| **Vocab size** | 50,257 (BPE) | ~37,000 (BPE, EN-DE) |
| **Context length** | 256 (training) | 512 |
| **Total Parameters** | ~124M | ~65M (base) / ~213M (big) |
| **Optimizer** | AdamW (lr=4e-4, wd=0.1) | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| **LR schedule** | Fixed | Warmup + inverse-sqrt decay |
| **Weight tying** | Yes | Yes |
| **Cross-attention** | No (decoder-only) | Yes (decoder attends to encoder) |
| **Label smoothing** | Not used | 0.1 |
| **Task** | Language modeling | Machine translation |

---

## Key Differences Explained

### 1. Architecture — Decoder-Only vs Encoder-Decoder

The original Transformer has two parts:
- **Encoder:** reads the input sentence (bidirectional — every token sees every other token)
- **Decoder:** generates the output sentence, attending to the encoder's output via cross-attention

GPT-2 removes the encoder entirely. There is only a decoder, and it is strictly **causal** — each token can only see itself and past tokens, never future ones. This makes it suitable for language modeling (predicting the next word) rather than translation.

### 2. GELU vs ReLU

The original paper uses ReLU in the feedforward network. GPT-2 uses GELU (Gaussian Error Linear Unit). GELU is smoother — it doesn't hard-zero negative values the way ReLU does — and empirically performs better for language models.

### 3. Pre-LN vs Post-LN

The original paper applies LayerNorm **after** the attention/FFN block and residual addition (Post-LN). GPT-2 applies it **before** (Pre-LN). Pre-LN training is more stable — gradients flow more cleanly through the residual path, reducing the risk of exploding or vanishing gradients during deep training.

### 4. Positional Encoding

The original paper uses a fixed mathematical formula (sinusoids) — no learned parameters at all. GPT-2 uses a learned embedding table (`nn.Embedding`), which the model trains alongside everything else. Learned embeddings often outperform sinusoidal ones in practice.

### 5. An Interesting Coincidence

Despite different model sizes and different numbers of heads:
```
GPT-2:  d_model / n_heads = 768 / 12 = 64
Paper:  d_model / n_heads = 512 / 8  = 64
```
Both arrive at the same head dimension (d_k = 64). This is not a coincidence in design philosophy — 64 dimensions per head has proven to be a sweet spot empirically.

---

## The Design Evolution

The GPT-2 implementation reflects improvements validated by the research community between 2017 and 2019:

| Original (2017) | Improved (GPT-2, 2019) | Why |
|---|---|---|
| ReLU | GELU | Better empirical performance |
| Post-LN | Pre-LN | More stable training |
| Sinusoidal PE | Learned PE | Adaptable to data |
| Adam | AdamW | Weight decay applied correctly |
| No dropout strategy | `drop_rate=0.1` systematically | Regularization built in |

---

## Related

- [[gpt2-from-scratch]]
- [[attention-is-all-you-need]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[layer-normalization]]
- [[positional-embeddings]]
- [[cross-attention]]
