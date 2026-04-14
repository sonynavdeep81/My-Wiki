---
title: Decoder Architecture (Slide Deck)
type: source
tags: [llm, transformer, decoder, tokenization, attention, embeddings]
sources: 1
updated: 2026-04-13
---

## Decoder Architecture (Slide Deck)

**Summary**: A comprehensive slide-by-slide walkthrough of LLM fundamentals — from what an LLM is, through tokenization, embeddings, and the full decoder-only transformer architecture with worked numerical examples.

## Key Points

- LLMs are trained on next-token prediction; [[emergent-abilities]] (reasoning, math, coding) arise automatically at scale
- In 2025, industry shifted from parameter scaling to [[inference-scaling]] (more compute per query)
- Three transformer variants: Encoder-Decoder (translation), Decoder-only (generation, GPT), Encoder-only (classification, BERT)
- Modern LLMs all use **Pre-Layer Normalization** (normalize before sublayer); original "Attention is All You Need" used Post-LN which is harder to train
- [[byte-pair-encoding]] (BPE) is the universal tokenization method; modern LLMs use 50k–100k vocab sizes
- GPT-2 embedding table: 50,257 × 768; positional embeddings are learnable (context size × 768)
- Attention: Q, K, V matrices computed via learned weight matrices (W_Q, W_K, W_V); 12 heads of dim 64 each for d_model=768
- Causal mask sets future positions to −∞ before softmax, preventing look-ahead
- Dropout (p=0.1) applied to attention weight matrix during training only; remaining values scaled by 1/(1−p)
- Contextual vector C = softmax(Q·K^T / √d_k) · V; heads concatenated and projected by W_O (768×768)
- FFN: 768 → 3072 (GELU) → 768; processes each token independently
- **Training**: teacher forcing — all positions computed in parallel using causal mask
- **Inference**: sequential (one token appended per step); [[kv-caching]] stores K/V to avoid recomputation
- Transformer block (Nx=12 for GPT-2): early blocks → syntax; middle → semantics; late → reasoning/world knowledge
- Linear layer projects 768 → 50,257 logits; softmax gives probability distribution; only last row used at inference
- GPT-3 training cost ~$4.6M; pre-trained models are called foundational models
- Open-source models (LLaMA 3 405B) now approach closed-source (GPT-4) performance on MMLU

## Concepts Found

- [[large-language-models]]
- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[tokenization]]
- [[byte-pair-encoding]]
- [[embeddings]]
- [[positional-embeddings]]
- [[layer-normalization]]
- [[multi-head-attention]]
- [[feed-forward-network]]
- [[scaling-laws]]
- [[emergent-abilities]]
- [[kv-caching]]
- [[inference-scaling]]

## Entities Found

- [[attention-is-all-you-need]] (paper)
- [[tiktoken]] (tool)
- [[llama]] (model family)
- [[gpt-family]] (model family)

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
