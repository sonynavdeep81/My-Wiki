---
title: Build a Large Language Model (From Scratch)
type: source
tags: [llm, gpt2, pretraining, fine-tuning, attention, tokenization, raschka]
sources: 1
updated: 2026-05-07
---

## Build a Large Language Model (From Scratch)

**Summary**: Full book by Sebastian Raschka (Manning, 2025) implementing GPT-2 from scratch in PyTorch across 7 chapters — covering tokenization, attention, architecture, pretraining, and both classification and instruction fine-tuning.

**Note**: This is the parent source for `gpt2_decoder.py`, `classification_fine_tuning.py`, and `instruction_fine_tuning.py` already in the wiki. Those files are code extracts from chapters 5–7.

## Book Structure

| Chapter | Topic |
|---|---|
| 1 | Understanding LLMs — next-token prediction, BERT vs GPT, zero/few-shot |
| 2 | Working with text data — tokenization, BPE, sliding window data loading |
| 3 | Attention mechanisms — simplified → trainable → causal → multi-head (4 variants) |
| 4 | GPT model architecture — transformer blocks, GELU, layer norm, shortcut connections |
| 5 | Pretraining — training loop, perplexity, save/load, load OpenAI weights, top-k/temp |
| 6 | Classification fine-tuning — spam classifier, freeze strategy, output head replacement |
| 7 | Instruction fine-tuning — Alpaca format, collate, LLM-based evaluation (AlpacaEval-style) |
| Appendix D | Training improvements — linear warmup + cosine decay + gradient clipping |

## GPT-2 Model Configurations

| Variant | Params | emb_dim | n_layers | n_heads |
|---|---|---|---|---|
| gpt2-small | 124M | 768 | 12 | 12 |
| gpt2-medium | 355M | 1024 | 24 | 16 |
| gpt2-large | 774M | 1280 | 36 | 20 |
| gpt2-xl | 1558M | 1600 | 48 | 25 |

## GPT-3 Training Dataset (Table 1.1)

| Dataset | Type | Tokens | Proportion |
|---|---|---|---|
| CommonCrawl (filtered) | Web crawl | 410B | 60% |
| WebText2 | Web crawl | 19B | 22% |
| Books1 | Internet books | 12B | 8% |
| Books2 | Internet books | 55B | 8% |
| Wikipedia | High-quality text | 3B | 3% |

Total available: 499B tokens; trained on: 300B (reason not specified by authors).

## Key Design Decisions Documented

- **Weight tying**: Raschka notes separate layers give better performance in practice; modern LLMs use separate layers
- **Instruction masking**: Book does NOT mask instruction tokens by default; Shi et al. 2024 shows unmasked instructions benefit performance
- **Gradient clipping**: Applied only *after* warmup phase, not during it
- **RLHF**: Mentioned as optional step post-instruction fine-tuning; not implemented in book

## Appendix D — Training Stabilization

Three techniques added to `train_model_simple`:
1. **Linear warmup**: LR rises from `initial_lr` (3e-5) to `peak_lr` over `warmup_steps`
2. **Cosine decay**: LR decays from `peak_lr` to `min_lr` (1e-6) following half-cosine curve
3. **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` applied after warmup phase only

## Key Points

- Book explicitly positions this as educational — uses small dataset; avoids full pretraining by loading OpenAI weights
- Attention progression in Ch. 3: 4 variants building from simple dot-product to full masked multi-head
- Evaluation: uses LLM-based scoring (Llama 3 as judge), inspired by AlpacaEval
- BERT referenced as contrast to GPT: encoder-only, masked prediction, used for classification

## Related

- [[gpt2-from-scratch]]
- [[multi-head-attention]]
- [[instruction-fine-tuning]]
- [[fine-tuning]]
- [[lr-warmup]]
- [[cosine-decay]]
- [[gradient-clipping]]
- [[sebastian-raschka]]
