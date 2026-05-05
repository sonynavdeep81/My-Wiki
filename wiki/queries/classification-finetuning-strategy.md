---
title: Classification Fine-Tuning Strategy — What to Freeze and What to Train
type: query
tags: [fine-tuning, classification, freezing, output-head, transformer-block, layer-normalization]
sources: 0
updated: 2026-04-24
---

## Classification Fine-Tuning Strategy — What to Freeze and What to Train

**Summary**: For classification fine-tuning, train only the final output head, final transformer block, and final LayerNorm; freeze all other 11 transformer blocks.

## What to Fine-Tune

| Component | Train/Freeze | Reason |
|---|---|---|
| Final output head (768→2) | **Train** | Brand new layer with random weights; replaces 50,257-class head |
| Final transformer block | **Train** | Adapts last-layer representations to classification task |
| Final LayerNorm | **Train** | Directly precedes output head; recalibrates hidden state for new task |
| All other 11 transformer blocks | **Freeze** | Already encode rich language understanding from pretraining |

## Why Freeze Most Layers

- Prevents **catastrophic forgetting** — pretrained language knowledge is preserved
- Drastically reduces **trainable parameters** — faster training
- Prevents **overfitting** on small classification datasets (e.g. SMS spam)
- Frozen layers already provide sufficient feature extraction for simple binary tasks

## Why the Output Head Must Always Be Trained

The pretrained GPT-2 output head maps 768 → 50,257 (next-token prediction). For classification it is **replaced** with a new 768 → num_classes head (e.g. 768→2 for spam/ham). This new head has random weights and has never seen data — it must always be trained regardless of strategy.

## Why Not Fine-Tune More Layers

For simple binary classification on short texts, the language understanding in frozen layers is more than sufficient. Fine-tuning just the top layers captures most of the performance gain at a fraction of the compute cost.

More layers can be unfrozen if:
- Task is complex or domain-specific
- Dataset is large enough to avoid overfitting
- Initial results show underfitting

## Related

- [[fine-tuning]]
- [[decoder-only-architecture]]
- [[layer-normalization]]
- [[gpt2-from-scratch]]
- [[spam-dataset-implementation]]
