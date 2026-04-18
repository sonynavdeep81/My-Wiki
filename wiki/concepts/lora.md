---
title: LoRA (Low-Rank Adaptation)
type: concept
tags: [fine-tuning, peft, lora, adaptation, low-rank]
sources: 1
updated: 2026-04-18
---

## LoRA (Low-Rank Adaptation)

**Summary**: Parameter-efficient fine-tuning method that freezes pretrained weights and injects trainable low-rank decomposition matrices into attention layers.

## Core Idea

Instead of updating W (d×d), decompose the update ΔW into two small matrices:

```
W' = W + ΔW = W + BA
where B ∈ R^{d×r}, A ∈ R^{r×d}, rank r << d
```

- W is frozen; only A and B are trained
- At init: A ~ N(0, σ²), B = 0  →  ΔW = 0 at start (no disruption)
- At inference: merge W + BA into single weight (zero latency overhead)

## Hyperparameters

| Param | Typical value | Effect |
|-------|--------------|--------|
| r (rank) | 4–64 | Controls capacity; higher r = more params |
| α (scaling) | r or 2r | Scales ΔW by α/r; effectively a lr multiplier |
| target modules | q_proj, v_proj | Which weight matrices get LoRA adapters |

Trainable params: 2 × r × d per layer (vs d² for full fine-tuning).

## Where to Apply (Placement)

Original paper: Q and V projections only.
Later work: also K, FFN layers — more coverage, more params.
**Key open question**: which layers matter most varies by task → see [[research-p3-sparse-lora]] (P3 paper).

## Variants

| Variant | Change |
|---------|--------|
| QLoRA | LoRA on 4-bit quantized base model; fits large models on 4GB VRAM |
| DoRA | Decomposes weight into magnitude + direction; LoRA on direction only |
| AdaLoRA | Adaptive rank allocation per layer based on importance score |
| Sparse LoRA | Prune A/B matrices; reduce active params further |

## Why LoRA Works

- Hypothesis: weight updates during fine-tuning have low intrinsic rank
- Evidence: r=4 matches full fine-tuning on many tasks at <0.1% of params
- Practical: enables fine-tuning 7B+ models on consumer GPUs

## Related

- [[fine-tuning]]
- [[gpt2-from-scratch]]
- [[research-p3-sparse-lora]]
- [[optimizer]]
