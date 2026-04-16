---
title: GPT-2 Implementation vs Attention Is All You Need — Parameter Comparison
type: query
tags: [comparison, gpt2, attention-is-all-you-need, hyperparameters]
updated: 2026-04-14
---

## GPT-2 Implementation vs Attention Is All You Need — Parameter Comparison

**Summary**: Side-by-side comparison of architecture and training hyperparameters between the user's GPT-2 decoder-only implementation and the original Transformer paper.

---

| Parameter | Your GPT-2 Implementation | Attention Is All You Need (base) |
|---|---|---|
| **Architecture** | Decoder-only | Encoder + Decoder |
| **d_model** | 768 | 512 |
| **d_ff** | 3072 (4×) | 2048 (4×) |
| **Attention heads (h)** | 12 | 8 |
| **d_k = d_v** | 64 | 64 |
| **Layers (N)** | 12 decoder layers | 6 encoder + 6 decoder |
| **Activation (FFN)** | GELU | ReLU |
| **Layer Norm placement** | Pre-LN | Post-LN |
| **Positional encoding** | Learnable (`nn.Embedding`) | Sinusoidal (fixed, no params) |
| **Dropout (P_drop)** | 0.1 | 0.1 |
| **Vocab size** | 50,257 (BPE) | ~37,000 (BPE, EN-DE) |
| **Context length** | 256 (training) | 512 |
| **Parameters** | ~124M | ~65M (base) / 213M (big) |
| **Optimizer** | AdamW (lr=0.0004, wd=0.1) | Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) |
| **LR schedule** | Fixed | Warmup + inverse-sqrt decay |
| **Weight tying** | Yes | Yes |
| **Cross-attention** | No (decoder-only) | Yes (decoder attends to encoder) |
| **Label smoothing** | Not noted | 0.1 |
| **Task** | Language modeling | Machine translation |

---

## Key Differences to Remember

**Structural**: The paper's encoder is bidirectional — every token sees every other token. Your GPT-2 is strictly causal throughout — every token only sees past tokens. The paper's decoder also has [[cross-attention]]; yours does not.

**Design choices**: Your implementation reflects post-2017 improvements — GELU over ReLU, Pre-LN over Post-LN, AdamW over Adam, learnable over sinusoidal positional embeddings. These are all changes validated by subsequent research.

**Coincidence**: Despite different d_model and head counts, both arrive at d_k=64 (`768/12 = 512/8 = 64`).

---

## Related

- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[attention-is-all-you-need]]
- [[Attention_2023|Attention Is All You Need (Paper)]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[layer-normalization]]
- [[positional-embeddings]]
- [[cross-attention]]
