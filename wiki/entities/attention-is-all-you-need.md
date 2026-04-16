---
title: Attention Is All You Need
type: entity
tags: [paper, transformer, vaswani, 2017]
sources: 1
updated: 2026-04-14
---

## Attention Is All You Need

**Summary**: The 2017 landmark paper by Vaswani et al. that introduced the Transformer architecture, replacing RNNs and convolutions with pure self-attention for sequence modeling.

- **Authors**: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google Brain / Google Research)
- **Venue**: NeurIPS 2017
- **ArXiv**: arXiv:1706.03762
- **Citations**: 174,000+

## Contributions

- Introduced the **Transformer** (encoder-decoder) for machine translation
- Showed self-attention alone — no recurrence, no convolutions — achieves SOTA
- Proposed **Scaled Dot-Product Attention** and **Multi-Head Attention**
- Introduced **sinusoidal positional encoding** (fixed, not learned)
- Used **Post-Layer Normalization** — later superseded by Pre-LN in practice
- Introduced LR warmup + inverse-sqrt decay schedule — now standard
- Applied **[[weight-tying]]** between input/output embeddings
- Three attention uses: encoder self-attention, decoder [[causal-masking|masked self-attention]], [[cross-attention]]

## Key Hyperparameters (base model)

`d_model=512, d_ff=2048, h=8, d_k=64, N=6, P_drop=0.1`

## FFN uses ReLU

The original paper's FFN uses ReLU: `FFN(x) = max(0, xW₁+b₁)W₂+b₂`. GPT-2 later switched to [[gelu|GELU]]. See [[feed-forward-network]].

## Impact

Every modern LLM is a variant of this architecture. GPT/LLaMA use the decoder-only variant (drop the encoder, keep only masked self-attention). T5/BART retain the full encoder-decoder with [[cross-attention]].

## Full Source

See [[Attention_2023]] for detailed notes on architecture, training, results, and ablations.

## Related

- [[transformer-architecture]]
- [[multi-head-attention]]
- [[cross-attention]]
- [[layer-normalization]]
- [[positional-embeddings]]
- [[feed-forward-network]]
- [[weight-tying]]
