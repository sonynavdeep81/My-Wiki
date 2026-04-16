---
title: Cross-Attention
type: concept
tags: [attention, cross-attention, encoder-decoder, transformer]
sources: 1
updated: 2026-04-14
---

## Cross-Attention

**Summary**: An attention sub-layer in the decoder where queries come from the decoder and keys/values come from the encoder output, allowing the decoder to attend to the full input sequence at every generation step.

---

## How It Differs from Self-Attention

In self-attention, Q, K, and V all come from the **same** sequence. In cross-attention:

```
Q  ←  decoder's current representation   (what am I looking for?)
K  ←  encoder's output                   (what does the input contain?)
V  ←  encoder's output                   (what do I retrieve?)
```

The computation is identical to self-attention:
`CrossAttention(Q, K, V) = softmax(QK^T / √d_k) · V`

The only difference is **where Q, K, V originate**.

---

## Role in the Encoder-Decoder Transformer

Each decoder layer has three sub-layers:

```
1. Masked Self-Attention     ← decoder attends to its own past outputs (causal)
2. Cross-Attention           ← decoder attends to full encoder output
3. Feed-Forward Network
```

Cross-attention is what enables sequence-to-sequence tasks: the decoder "reads" the encoded input representation at every step while generating the output.

---

## Three Uses of Attention in the Original Transformer

From [[Attention_2023|the paper]]:

| Type | Q source | K/V source | Mask |
|---|---|---|---|
| Encoder self-attention | Encoder layer output | Same | None (bidirectional) |
| Decoder self-attention | Decoder layer output | Same | Causal (upper-triangle −∞) |
| Cross-attention | Decoder layer output | Encoder final output | None |

---

## Decoder-Only Models Have No Cross-Attention

GPT-style ([[decoder-only-architecture]]) models drop the encoder entirely. There is no cross-attention — the model only has masked self-attention. This is sufficient for language modeling but requires a different approach for seq2seq tasks (e.g. conditioning on a prompt instead of a separate encoder).

Models that do use cross-attention: original Transformer, T5, BART, Whisper (audio encoder + text decoder).

---

## Related

- [[multi-head-attention]]
- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[attention-is-all-you-need]]
- [[causal-masking]]
