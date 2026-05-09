---
title: Large Language Models (LLMs)
type: concept
tags: [llm, foundation-model, scaling, multimodal]
sources: 2
updated: 2026-05-07
verified_against: Raschka-LLM-2025, 2026-05-07
confidence: high
---

## Large Language Models (LLMs)

**Summary**: Neural networks with billions–trillions of parameters trained to predict the next token, which incidentally learn language, reasoning, and world knowledge.

## What They Are

An LLM is a model trained to predict the next token in a sequence. Despite this simple objective, they learn grammar, translation, reasoning, math, and coding as [[emergent-abilities]].

All input modalities (audio, video, images) are internally converted to tokens — the core training goal remains next-token prediction. Multimodal abilities are layered on top without changing the core.

## Parameter Counts (Representative)

| Model | Parameters |
|---|---|
| GPT-2 | 1.5B |
| GPT-3 | 175B |
| LLaMA 3 | 405B |
| Mistral 7B | 7B |
| Claude 3 Opus | >200B (est.) |
| GPT-5.1 (Thinking) | ~2T (est.) |

More parameters generally means more capability, but [[scaling-laws]] govern the relationship between parameters, data, and compute.

## 2025 Paradigm Shift

In 2025, the industry shifted focus from **parameter scaling** (making models bigger) to **[[inference-scaling]]** (giving models more time to "think" per query).

## Open vs Closed Source

- **Closed-source** (GPT-5.1, Claude, Gemini): proprietary, higher peak performance, no public fine-tuning
- **Open-source** ([[llama]], Mistral, Mixtral): publicly released weights, allow fine-tuning and local deployment
- The performance gap is shrinking: LLaMA 3 405B nearly matches GPT-4 on MMLU

## Training

Pre-training is expensive (TPUs/GPUs, weeks of computation). GPT-3 cost ~$4.6M. Pre-trained models are called **foundational models**. See [[decoder-only-architecture]] for how training works.

## GPT-3 Training Data Breakdown

| Dataset | Type | Tokens | % sampled |
|---|---|---|---|
| CommonCrawl (filtered) | Web crawl | 410B | 60% |
| WebText2 | Web crawl | 19B | 22% |
| Books1 + Books2 | Internet books | 67B | 16% |
| Wikipedia | High-quality | 3B | 3% |

Trained on 300B of 499B available tokens (reason not specified).

## Generalization Abilities

LLMs generalize to unseen tasks via [[zero-shot-few-shot]] learning — no weight updates needed, only prompt construction. This is an [[emergent-abilities]] of scale.

## Related

- [[transformer-architecture]]
- [[emergent-abilities]]
- [[scaling-laws]]
- [[inference-scaling]]
- [[tokenization]]
- [[zero-shot-few-shot]]
