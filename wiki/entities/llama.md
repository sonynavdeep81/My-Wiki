---
title: LLaMA (Model Family)
type: entity
tags: [model, open-source, meta, llm]
sources: 1
updated: 2026-04-13
---

## LLaMA (Model Family)

**Summary**: Meta's family of open-source large language models, with LLaMA 3 405B nearly matching GPT-4 performance on MMLU benchmarks.

- **Developer**: Meta AI
- **Type**: Open-source, decoder-only transformer
- **License**: Publicly released weights (with usage restrictions)

## Key Models

| Model | Parameters | Notes |
|---|---|---|
| LLaMA 1 65B | 65B | First major open-source competitor |
| LLaMA 2 70B | 70B | Improved alignment |
| LLaMA 3 8B | 8B | Efficient small model |
| LLaMA 3 70B | 70B | Strong mid-size |
| LLaMA 3 405B | 405B | Near GPT-4 level on MMLU; major milestone |

## Significance

LLaMA 3 405B is a landmark in the **open vs closed-source** race — it nearly matches GPT-4 on MMLU, significantly narrowing the performance gap. Enables:
- Domain-specific fine-tuning (e.g., legal, medical)
- Local/private deployment (no data sent to third-party APIs)
- Cost-effective high-volume inference via quantized models

## Related

- [[large-language-models]]
- [[decoder-only-architecture]]
- [[scaling-laws]]
