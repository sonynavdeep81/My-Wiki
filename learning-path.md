---
title: Learning Path
type: reference
updated: 2026-05-05
---

# Learning Path — LLM Internals & NLP

Canonical reading order. Each stage builds on the previous.

---

## Stage 1: Foundations

- [[Large Language Models]]
- [[Zero-Shot and Few-Shot Learning]]
- [[Tokenization]]
- [[Byte-Pair Encoding]]
- [[Embeddings]]
- [[Positional Embeddings]]

## Stage 2: Transformer Internals

- [[Transformer Architecture]]
- [[BERT]]
- [[Softmax]]
- [[Multi-Head Attention]]
- [[Cross-Attention]]
- [[Causal Masking]]
- [[Feed-Forward Network]]
- [[GELU Activation]]
- [[Layer Normalization]]
- [[Residual Connections]]
- [[Dropout]]

## Stage 3: GPT-2 Architecture

- [[Decoder-Only Architecture]]
- [[GPT-2 From-Scratch Implementation Patterns]]
- [[Weight Tying]]
- [[PyTorch NN Building Blocks]]

## Stage 4: Training Mechanics

- [[Cross-Entropy Loss]]
- [[Optimizer]]
- [[LR Warmup]]
- [[Cosine Decay]]
- [[Gradient Clipping]]
- [[Label Smoothing]]

## Stage 5: Inference & Decoding

- [[KV Caching]]
- [[Decoding Strategies]]
- [[Temperature]]
- [[Inference Scaling]]

## Stage 6: Fine-Tuning & Adaptation

- [[Fine-Tuning]]
- [[Instruction Fine-Tuning]]
- [[LoRA]]

## Stage 7: Evaluation & Scaling

- [[Perplexity]]
- [[BLEU Score]]
- [[Scaling Laws]]
- [[Emergent Abilities]]

---

## Gaps in This Path

Topics that belong in the sequence but have no concept page yet:

- **Stage 2**: Sparse attention, relative positional encodings (RoPE, ALiBi)
- **Stage 4**: Mixed precision training (fp16/bf16)
- **Stage 5**: Beam search internals, speculative decoding
- **Stage 6**: RLHF, DPO, QLoRA, prefix tuning
- **Stage 7**: ROUGE score, MT-Bench, MMLU evaluation
