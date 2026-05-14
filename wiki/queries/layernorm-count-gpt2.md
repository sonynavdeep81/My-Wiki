---
title: How Many LayerNorm Layers Does GPT-2 Have?
type: query
tags: [layer-normalization, gpt-2, parameter-count, architecture]
sources: 2
updated: 2026-05-14
---

## How Many LayerNorm Layers Does GPT-2 Have?

**Summary**: GPT-2 124M has **25 LayerNorm layers** — two inside each of the 12 transformer blocks (Pre-LN: one before attention, one before FFN), plus one final LayerNorm before the output head. Each carries its own independent `(γ, β)` pair of shape `(768,)`. Together they account for `25 × 2 × 768 = 38,400` learnable parameters. Each block has its own LayerNorms because each block sits at a different depth and processes a different distribution.

---

## The Common Mistake

A natural but wrong answer:

> "GPT-2 has three LayerNorms — two inside the transformer block and one final."

That is the count for a **single transformer block** plus the final norm. GPT-2 124M has **12 transformer blocks**, so you must multiply.

| Model | Blocks (`n_layers`) | LayerNorms |
|---|---|---|
| GPT-2 124M (small) | 12 | 12 × 2 + 1 = **25** |
| GPT-2 355M (medium) | 24 | 24 × 2 + 1 = **49** |
| GPT-2 774M (large) | 36 | 36 × 2 + 1 = **73** |
| GPT-2 1.5B (XL) | 48 | 48 × 2 + 1 = **97** |

The general formula:

```
total_layernorms = 2 × n_layers + 1
```

---

## Where the LayerNorms Sit (Pre-LN)

GPT-2 uses the **Pre-LN** ordering — LayerNorm runs *before* the sublayer, then the sublayer's output is added back to the input via a residual connection.

```
Input: token_emb + pos_emb     # (batch, seq_len, 768)
│
├── Transformer Block 1
│     │
│     ├── x_norm = LayerNorm₁(x)              ← γ₁, β₁
│     ├── attn_out = MultiHeadAttention(x_norm)
│     ├── x = x + attn_out                     ← residual
│     │
│     ├── x_norm = LayerNorm₂(x)              ← γ₂, β₂
│     ├── ffn_out = FeedForward(x_norm)
│     └── x = x + ffn_out                      ← residual
│
├── Transformer Block 2
│     ├── LayerNorm₃    ← γ₃, β₃
│     └── LayerNorm₄    ← γ₄, β₄
│
... (10 more blocks, 20 more LayerNorms: LN₅ ... LN₂₄)
│
├── Transformer Block 12
│     ├── LayerNorm₂₃   ← γ₂₃, β₂₃
│     └── LayerNorm₂₄   ← γ₂₄, β₂₄
│
└── Final LayerNorm₂₅   ← γ₂₅, β₂₅   (the "ln_f" in OpenAI checkpoints)
   │
   └── out_head: Linear(768, 50257)   # produces logits
```

**Two per block.** Per the [[layer-normalization]] Pre-LN pattern, every sublayer (attention and FFN) is preceded by its own LayerNorm. Two sublayers per block → two LayerNorms per block.

**One final.** After the last block, one more LayerNorm normalizes the final hidden states before they are projected to vocabulary logits. In the OpenAI GPT-2 checkpoint this is named `ln_f` ("LayerNorm final").

---

## Parameter Count Contribution

Each LayerNorm has only `2 × 768 = 1536` params (one `γ` and one `β` per feature, see [[layernorm-scale-shift-sharing]]).

For GPT-2 124M:

```
25 LayerNorms × 1536 params  =  38,400 params total
```

That is roughly **0.03%** of the model's 124M parameters — tiny in absolute count, but architecturally critical for trainability.

| Model | LayerNorms | LayerNorm params | % of total |
|---|---|---|---|
| GPT-2 124M | 25 | 38,400 | 0.03% |
| GPT-2 355M | 49 | 49 × 2 × 1024 = 100,352 | 0.03% |
| GPT-2 774M | 73 | 73 × 2 × 1280 = 186,880 | 0.02% |
| GPT-2 1.5B | 97 | 97 × 2 × 1600 = 310,400 | 0.02% |

The LayerNorm contribution stays roughly flat as a percentage because it scales linearly with `n_layers × emb_dim`, while total params scale roughly with `n_layers × emb_dim²`.

---

## Why Each Block Gets Its Own LayerNorm — Why Not Share?

A natural follow-up: if `(γ, β)` are shared across all tokens within one LayerNorm, why not also share across blocks? It would save params.

**Answer: each block sees a different distribution.**

- Block 1 receives the raw `token_emb + pos_emb` — embeddings of independent tokens, distribution shaped by the embedding initialization.
- Block 6 receives heavily attention-mixed representations — distribution is shaped by the attention patterns of blocks 1–5.
- Block 12 receives deeply transformed, near-output representations — distribution is shaped to support the final classification into 50,257 vocabulary buckets.

These distributions have different means, variances, and *useful directions in feature space*. Forcing one shared `(γ, β)` across all 25 LayerNorms would be a one-size-fits-all affine that suits no layer well — a ceiling on capacity for negligible param savings (38K out of 124M).

The same logic explains why each block has its own attention weights, FFN weights, etc. Depth-specific parameters allow the network to learn depth-specific abstractions.

---

## Verifying in Code

You can confirm the count directly from a built model:

```python
import torch.nn as nn
from your_model import GPTModel, GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)

ln_layers = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
print(f"Number of LayerNorms: {len(ln_layers)}")        # 25

ln_params = sum(p.numel() for m in ln_layers for p in m.parameters())
print(f"LayerNorm params:      {ln_params:,}")          # 38,400
```

If your custom LayerNorm class is not `nn.LayerNorm`, swap the type. For Raschka's `LayerNorm` (a hand-rolled version):

```python
from your_model import LayerNorm
ln_layers = [m for m in model.modules() if isinstance(m, LayerNorm)]
```

---

## Related

- [[layer-normalization]] — the concept page
- [[layernorm-scale-shift-sharing]] — why `(γ, β)` are shared across tokens but not across layers
- [[gpt2-parameter-count]] — full breakdown of GPT-2 124M's 124M (or 162M when tied weights are double-counted) parameters
- [[gpt2-from-scratch]] — implementation patterns including Pre-LN ordering
- [[residual-connections]] — the skip pattern that pairs with Pre-LN
