---
title: GPT Family (OpenAI)
type: entity
tags: [model, openai, gpt, closed-source, llm]
sources: 1
updated: 2026-04-13
---

## GPT Family (OpenAI)

**Summary**: OpenAI's series of decoder-only transformer models, from GPT-2 (open weights) through GPT-3/4/5 (closed API), that established the decoder-only architecture as the dominant paradigm for LLMs.

## Key Models

| Model | Parameters | Notes |
|---|---|---|
| GPT-2 | 1.5B | Open weights; used as reference implementation |
| GPT-3 | 175B | $4.6M to train; established few-shot prompting |
| GPT-4 | Unknown | Multimodal; ~1.8T MoE (estimated) |
| GPT-5.1 (Instant) | ~200B est. | Fast inference variant |
| GPT-5.1 (Thinking) | ~2T est. | [[inference-scaling]] model |

## GPT-2 Architecture Details

- Vocab: 50,257 tokens (BPE via [[tiktoken]])
- Embedding dim: 768
- Heads: 12, Layers: 12
- Context length: 1024
- Uses **learnable positional embeddings** (not sinusoidal)
- Uses **Pre-LN** (Post-LN in original Transformer paper)
- **qkv_bias=True** in OpenAI checkpoint (differs from modern practice)
- Weight tying between `tok_emb` and `out_head` (see [[weight-tying]])
- OpenAI stores Q, K, V concatenated as `c_attn` (shape: 768×2304); must split to load

## GPT-3 Training Data

| Dataset | Tokens | Weight |
|---|---|---|
| Common Crawl (filtered) | 410B | 60% |
| WebText2 | 19B | 22% |
| Books1 | 12B | 8% |
| Books2 | 55B | 8% |
| Wikipedia | 3B | 3% |

Total: ~300B tokens seen; cost ~$4.6M.

## Related

- [[large-language-models]]
- [[decoder-only-architecture]]
- [[gpt2-from-scratch]]
- [[tiktoken]]
- [[weight-tying]]
- [[inference-scaling]]
