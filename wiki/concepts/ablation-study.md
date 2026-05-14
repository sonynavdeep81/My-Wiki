---
title: Ablation Study
type: concept
tags: [research-methodology, evaluation, experimental-design, attribution, paper-writing]
sources: 0
updated: 2026-05-14
confidence: high
---

## Ablation Study

**Summary**: Remove one component of a system at a time and measure the performance drop. Quantifies each component's contribution to the result and forces honest attribution in research papers.

## Origin

| Term | Source | Meaning |
|---|---|---|
| ablation | Latin *ablatio* (removal) — surgical | ML/research: removing a component to measure its contribution |

## Mechanics

1. Take the full system → measure metric `M_full`
2. Remove component `X` → measure `M_{-X}`
3. Drop = `M_full − M_{-X}`
4. Repeat for every component you claim matters

## Output: Ablation Table

| Configuration | Metric | Δ vs full |
|---|---|---|
| Full model | M_full | — |
| Minus A | M_{-A} | M_full − M_{-A} |
| Minus B | M_{-B} | M_full − M_{-B} |
| ... | ... | ... |

- Big drop → component earned its place
- ≈ 0 drop → candidate for removal from the paper

## Why It Is Mandatory in a Paper

| Reason | What ablation prevents |
|---|---|
| Justify each design choice | "Why did you add X?" → answered with numbers |
| Prevent false credit | Stops attributing gains to the wrong component |
| Honest attribution | Distinguishes architecture wins from data/scale wins |
| Field simplification | Removes decoration components from future work |

## What Ablation Is NOT

| Activity | Why it is not ablation |
|---|---|
| Hyperparameter tuning | Tunes a knob; does not remove a component |
| Baseline comparison | Compares whole models; not full-vs-(full−X) |
| Architecture search | Adds components; subtraction is the inverse |
| Sensitivity analysis | Varies a value; ablation removes/disables outright |

## Types

| Type | Example |
|---|---|
| Component | Remove a module / layer / sublayer |
| Loss | Remove auxiliary loss term (e.g., NSP in BERT) |
| Data | Train on a subset of data sources |
| Architecture | Replace component with a simpler operator (attention → identity) |
| Hyperparam-bounded | Set `drop_rate=0`, `warmup_steps=0`, etc. |

## Famous Examples

| Paper | Ablated | What was learned |
|---|---|---|
| [[Attention Is All You Need]] (Vaswani 2017) Table 3 | `n_heads`, `d_model`, dropout, PE type | sinusoidal ≈ learned PE; 8 heads ≫ 1 head |
| BERT (Devlin 2019) | NSP loss, MLM masking ratio | NSP adds little; 15% mask rate is a sweet spot |
| Chinchilla (Hoffmann 2022) | model size vs data size at fixed compute | optimal ratio: ~20 tokens per param |
| LoRA (Hu 2021) | which weight matrices to adapt | adapting Q+V is enough; full FT not needed |

## Application to GPT-2 124M (your build)

Suggested ablation rows for `gpt2_decoder.py`:

| Variant | What you measure |
|---|---|
| Full model | val loss baseline |
| Without [[weight-tying]] | how much WT helps convergence + param count |
| Without [[dropout]] (drop_rate=0) | overfitting amount on small dataset |
| Without LayerNorm | usually diverges (canonical ablation) |
| Without residuals | usually diverges (canonical ablation) |
| 6 blocks vs 12 | depth contribution |
| `qkv_bias=False` | bias contribution under LayerNorm |

See [[ablation-study-explained]] for the full pedagogical walkthrough with examples.

## Related

- [[llm-evaluation-metrics]]
- [[scaling-laws]]
- [[bias-comparison-gpt2-vs-paper]]
- [[gpt2-vs-attention-paper-params]]
- [[fine-tuning]]
- [[lora]]
