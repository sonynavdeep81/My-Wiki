---
title: Attention Is All You Need
type: entity
tags: [paper, transformer, vaswani, 2017]
sources: 1
updated: 2026-04-13
---

## Attention Is All You Need

**Summary**: The 2017 landmark paper by Vaswani et al. that introduced the Transformer architecture, replacing RNNs with self-attention for sequence modeling.

- **Authors**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, et al.
- **Venue**: Advances in Neural Information Processing Systems 30 (NeurIPS 2017)
- **Citations**: 174,521+
- **ArXiv**: https://arxiv.org/pdf/1706.03762

## Contributions

- Introduced the **Transformer** architecture (encoder-decoder) for machine translation
- Demonstrated that self-attention alone (no recurrence, no convolutions) is sufficient
- Used **Post-Layer Normalization** (Sublayer → Add → Norm) — later superseded by Pre-LN in practice
- Enabled parallelization across sequence positions during training

## Impact

Every modern LLM is built on (a variant of) this architecture. The decoder-only variant (used by GPT, LLaMA, Mistral) drops the encoder and uses only the decoder stack with causal masking.

## Related

- [[transformer-architecture]]
- [[multi-head-attention]]
- [[layer-normalization]]
