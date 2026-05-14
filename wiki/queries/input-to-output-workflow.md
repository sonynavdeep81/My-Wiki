---
title: Complete Workflow — Input Text to Output Token
type: query
tags: [workflow, inference, tokenization, attention, decoding, end-to-end, dropout, residual]
sources: 2
updated: 2026-05-14
---

## Complete Workflow — Input Text to Output Token

**Summary**: End-to-end walkthrough from raw input text to a generated next token in GPT-2, including shapes, dropout placements, and residual connections at every stage.

---

## Overview

GPT-2 takes a sequence of text, converts it to numbers, passes those numbers through 12 transformer blocks, and produces a probability distribution over its 50,257-word vocabulary. It then samples from that distribution to pick the next word. This process repeats until the full output is generated.

---

## Stage 1 — Tokenization

```
Input text:  "Every effort takes you"
             ↓
Token IDs:   [464, 3797, 3332, 319]   shape: (4,)
```

The text is converted to integer IDs using tiktoken's BPE tokenizer (the same one OpenAI used for GPT-2). Each word or sub-word maps to a number in the 50,257-word vocabulary.

---

## Stage 2 — Embedding Layer

```
Token IDs: [464, 3797, 3332, 319]   shape: (4,)
           ↓
tok_emb:   shape (4, 768)    ← each ID looked up in 50,257×768 table
pos_emb:   shape (4, 768)    ← positions [0,1,2,3] looked up in 256×768 table
           ↓
x = tok_emb + pos_emb        shape: (4, 768)
x = Dropout(0.1)(x)          shape: (4, 768)  ← training only
```

Each token gets two embeddings added together:
- **Token embedding:** what is this token? (meaning)
- **Positional embedding:** where is this token in the sequence? (position)

Dropout is applied once here during training to regularize the embedding layer.

---

## Stage 3 — 12 Transformer Blocks (Repeated)

Each of the 12 blocks does the same two operations in sequence, both with residual connections:

### Attention Sub-Block

```
shortcut = x                              ← save input for residual
x = LayerNorm(x)                          ← normalize before attention
x = MultiHeadAttention(x)                 ← tokens attend to each other
x = Dropout(0.1)(x)                       ← training only
x = x + shortcut                          ← residual connection 1
```

Inside MultiHeadAttention:
- $Q = x W_Q$, $K = x W_K$, $V = x W_V$ — project to queries, keys, values
- Split into 12 heads, each with dimension 64 ($768 / 12 = 64$)
- $\text{att\_scores} = Q K^{\top} / \sqrt{64}$ — scaled dot-product attention
- Apply causal mask: set future positions to $-\infty$ (after softmax these become $0$)
- $\text{att\_weights} = \text{softmax}(\text{att\_scores}) \to$ apply dropout on weights
- $\text{context} = \text{att\_weights} \cdot V \to$ concat 12 heads $\to$ project via $W_O$

### FeedForward Sub-Block

```
shortcut = x                              ← save input for residual
x = LayerNorm(x)                          ← normalize before FFN
x = FeedForward(x)                        ← 768 → 3072 → 768
x = Dropout(0.1)(x)                       ← training only
x = x + shortcut                          ← residual connection 2
```

The FFN processes each token independently (no token mixing here — all mixing happens in attention). GELU activation between the two linear layers.

---

## Stage 4 — Final LayerNorm and Output Head

```
x = LayerNorm(x)              shape: (4, 768)
logits = out_head(x)          shape: (4, 50257)
```

The output head maps each token's 768-dim vector to 50,257 scores — one per vocabulary word. This gives a score for "what comes next after this token?"

---

## Stage 5 — Decoding (Inference Only)

Only the **last token's** scores are used at inference time:

```
logits[-1]           shape: (50257,)   ← scores for "what comes after 'you'?"
         ↓
top-k masking        zero out all but top-25 scores
         ↓
÷ temperature (1.4)  flatten the distribution (higher = more random)
         ↓
softmax              convert scores to probabilities
         ↓
multinomial sample   randomly pick one token from the distribution
         ↓
token ID: 2651       → decoded to "forward"
```

---

## Full Shape Trace

| Stage | Shape | Note |
|---|---|---|
| Token IDs | (4,) | integer token indices |
| After embedding | (4, 768) | tok_emb + pos_emb |
| After each block | (4, 768) | shape unchanged through all 12 blocks |
| After final norm | (4, 768) | |
| Logits | (4, 50257) | all positions |
| Inference logits | (50257,) | last row only |
| Sampled token | scalar | one new token ID |

---

## Dropout Placements — 3 Per Block

| Location | When active |
|---|---|
| After embedding (tok + pos) | Training only |
| After attention weights (inside MHA) | Training only |
| After MHA output (before residual) | Training only |
| After FFN output (before residual) | Training only |

All disabled automatically at inference via `model.eval()`.

---

## Full Pipeline Diagram

```
"Every effort takes you"
         │
    TOKENIZATION
    tiktoken BPE
         │ [464, 3797, 3332, 319]
         ▼
    EMBEDDING LAYER
    tok_emb + pos_emb + Dropout
         │ (4, 768)
         ▼
    ┌─────────────────┐
    │  REPEAT ×12     │
    │                 │
    │  LayerNorm      │
    │  ↓              │
    │  MultiHead      │
    │  Attention      │
    │  + causal mask  │
    │  + Dropout      │
    │  + residual     │
    │                 │
    │  LayerNorm      │
    │  ↓              │
    │  FeedForward    │
    │  768→3072→768   │
    │  + Dropout      │
    │  + residual     │
    └────────┬────────┘
             │ (4, 768)
         FINAL NORM
             │
         OUT HEAD
         768 → 50257
             │ (4, 50257)
    last row only ↓
         DECODING
         top-k → temperature → softmax → sample
             │
         "forward"
```

---

## Key Facts to Remember

- **Training uses all positions:** during training, the loss is computed on all 4 positions simultaneously (teacher forcing). Inference uses only the last position.
- **Causal mask enables parallel training:** because future tokens are masked, the model can be trained on all positions at once — no need to run forward passes sequentially.
- **FFN has no token mixing:** all communication between tokens happens in attention. The FFN is applied independently to each token's vector.
- **Residual connections are critical:** they allow gradients to flow directly through the network without passing through attention or FFN. Without them, training 12+ layers would be very difficult.

---

## Related

- [[tokenization]]
- [[embeddings]]
- [[multi-head-attention]]
- [[decoder-only-architecture]]
- [[decoding-strategies]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[dropout]]
- [[causal-masking]]
- [[residual-connections]]
- [[kv-caching]]
