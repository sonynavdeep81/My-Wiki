---
title: Attention Is All You Need (Paper)
type: source
tags: [transformer, attention, paper, vaswani, 2017]
updated: 2026-04-14
---

## Attention Is All You Need (Paper)

**Summary**: The 2017 foundational paper by Vaswani et al. (Google Brain) that introduced the Transformer architecture — replacing RNNs and convolutions entirely with self-attention — and set SOTA on machine translation.

- **Authors**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
- **Venue**: NeurIPS 2017
- **ArXiv**: arXiv:1706.03762v7 (2023 reprint)

---

## Architecture Overview

Encoder-decoder with N=6 layers each. Each layer has two sub-layers (encoder) or three (decoder):

```
Encoder layer:       Multi-Head Self-Attention → FFN
Decoder layer:       Masked Multi-Head Self-Attention → Cross-Attention → FFN
```

All sub-layers use: **residual connection + Post-LN** → `LayerNorm(x + Sublayer(x))`

Key hyperparameters (base model):

| Param | Value |
|---|---|
| d_model | 512 |
| d_ff | 2048 (4×) |
| h (heads) | 8 |
| d_k = d_v | 64 |
| N (layers) | 6 |
| P_drop | 0.1 |

---

## Key Technical Contributions

### Scaled Dot-Product Attention
`Attention(Q,K,V) = softmax(QK^T / √d_k) · V`

Scaling by √d_k prevents softmax saturation when d_k is large (dot products grow in magnitude → extremely small gradients).

### Multi-Head Attention
Project Q, K, V h=8 times into d_k=64 subspaces, compute attention in parallel, concatenate:
`MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O`

Allows the model to attend to different representation subspaces simultaneously. See [[multi-head-attention]].

### Three Uses of Attention
1. **Encoder self-attention** — every position attends to all positions in previous encoder layer (bidirectional)
2. **Decoder self-attention** — causal/masked; each position attends only to previous positions
3. **Cross-attention** — decoder queries attend to encoder keys/values; see [[cross-attention]]

### FFN Uses ReLU (not GELU)
`FFN(x) = max(0, xW₁+b₁)W₂+b₂`

GPT-2 later replaced ReLU with [[gelu|GELU]]. See [[feed-forward-network]].

### Sinusoidal Positional Encoding
Fixed, not learned:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Hypothesized to allow extrapolation to unseen sequence lengths. Learned embeddings performed identically in ablations (Table 3, row E). See [[positional-embeddings]].

### Post-LN (vs modern Pre-LN)
The paper uses Post-LN: normalize *after* the residual add. This requires careful LR warmup. Modern models (GPT-2/3, LLaMA) use Pre-LN for easier training. See [[layer-normalization]].

### Weight Tying
Input/output embeddings and the pre-softmax linear layer share the same matrix, weighted by √d_model. Same trick used in GPT-2. See [[weight-tying]].

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) |
| LR schedule | Warmup then inverse sqrt decay |
| Warmup steps | 4000 |
| Regularization | Residual dropout (P=0.1), label smoothing (ε=0.1) |
| Hardware | 8× NVIDIA P100 |
| Base training | 100K steps (~12 hours) |
| Big training | 300K steps (~3.5 days) |

**LR formula**: `lrate = d_model^-0.5 · min(step^-0.5, step · warmup_steps^-1.5)`

**Label Smoothing** (ε=0.1): instead of a hard 0/1 target, distribute 0.1 probability mass across all tokens. Hurts perplexity but improves BLEU.

---

## Results

| Task | Model | BLEU |
|---|---|---|
| EN→DE | Transformer (big) | **28.4** (prior SOTA: ~26.4) |
| EN→FR | Transformer (big) | **41.8** (prior SOTA: ~41.3) |

Achieved at a fraction of the training FLOPs of prior models.

---

## Why Self-Attention vs RNN/CNN

| Layer Type | Complexity/layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |

Self-attention connects any two positions in O(1) steps — critical for learning long-range dependencies. RNNs require O(n) sequential steps to connect distant positions.

---

## Attention Head Specialization (Appendix)

Visualizations show heads learn distinct roles:
- Long-distance syntactic dependencies (e.g. "making...difficult")
- Anaphora resolution (e.g. "its" → "application")
- Sentence structure patterns

Different heads at the same layer clearly learn different linguistic tasks.

---

## Related

- [[attention-is-all-you-need]]
- [[multi-head-attention]]
- [[cross-attention]]
- [[transformer-architecture]]
- [[positional-embeddings]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[weight-tying]]
- [[dropout]]
