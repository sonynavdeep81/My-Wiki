---
title: Transformer Architecture
type: concept
tags: [transformer, attention, neural-network, architecture]
sources: 1
updated: 2026-04-13
---

## Transformer Architecture

**Summary**: A deep neural network architecture introduced in "Attention is All You Need" (2017) that uses self-attention to process all tokens in parallel, now the foundation of all modern LLMs.

## Origin

Introduced in [[attention-is-all-you-need]] (Vaswani et al., 2017), originally for machine translation (English → German/French). Uses an **Encoder-Decoder** structure: the encoder reads the source, the decoder generates the target one word at a time.

## Three Variants

| Variant | Use Case | Examples |
|---|---|---|
| Encoder-Decoder | Translation, seq2seq | BART, T5 |
| Decoder-only | Text generation, summarization | GPT-3, LLaMA, Mistral |
| Encoder-only | Classification, embeddings | BERT |

Modern [[large-language-models]] use the decoder-only variant. See [[decoder-only-architecture]].

## Self-Attention

Self-attention allows every token to attend to every other token in the input simultaneously. Advantages:
- Better long-range context understanding
- Parallelizable (unlike RNNs)
- Better handling of complex dependencies

## Layer Normalization: Pre vs Post

- **Post-LN** (original paper): Sublayer → Residual Add → Normalize. Harder to train; requires careful LR warmup.
- **Pre-LN** (modern standard): Normalize → Sublayer → Residual Add. Easier to train, performs just as well.

All modern models (GPT-3, LLaMA, BART, T5) use **Pre-LN**.

## Core Components (Decoder block)

1. Token + [[positional-embeddings]]
2. [[multi-head-attention]] (masked, causal)
3. Residual connection + [[layer-normalization]]
4. [[feed-forward-network]] (FFN)
5. Residual connection + Layer Norm
6. (After Nx blocks) Linear → Softmax

## Related

- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[attention-is-all-you-need]]
