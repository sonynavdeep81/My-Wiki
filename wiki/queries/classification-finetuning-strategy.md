---
title: Classification Fine-Tuning Strategy — What to Freeze and What to Train
type: query
tags: [fine-tuning, classification, freezing, output-head, transformer-block, layer-normalization]
sources: 0
updated: 2026-05-14
---

## Classification Fine-Tuning Strategy — What to Freeze and What to Train

**Summary**: For classification fine-tuning on GPT-2, train only the final output head, final transformer block, and final LayerNorm. Freeze all other 11 transformer blocks.

---

## The Core Idea

GPT-2 is pretrained on vast amounts of text and already understands language deeply. When you fine-tune it for classification (e.g., spam detection), you don't need to retrain the whole model. Most of its knowledge is already useful — you just need to redirect the final output towards your task.

Think of it like hiring an expert linguist to classify documents. You don't re-teach them English — you just tell them the specific categories to assign.

---

## What to Train and What to Freeze

| Component | Decision | Reason |
|---|---|---|
| Final output head (768 → 2) | **Train** | Brand new layer with random weights; must always be trained |
| Final transformer block (block 11) | **Train** | Adapts last-layer representations to the classification task |
| Final LayerNorm | **Train** | Directly precedes the output head; recalibrates hidden state for new task |
| All other 11 transformer blocks (0–10) | **Freeze** | Already encode rich language understanding; preserve pretrained knowledge |

---

## Why Freeze Most Layers?

**Prevents catastrophic forgetting:** If you train all layers on a small classification dataset, the model will overwrite its pretrained language knowledge. The frozen layers remember everything GPT-2 learned during pretraining.

**Reduces trainable parameters:** With 11 blocks frozen, you update only a small fraction of the 124M parameters. Training is much faster.

**Prevents overfitting:** Small datasets (like an SMS spam dataset with a few thousand examples) cannot support training 124M parameters. Freezing most layers keeps the model from memorizing training examples.

**Frozen layers are already useful:** The lower transformer blocks capture general language patterns — grammar, syntax, semantics. These transfer directly to classification tasks without any modification.

---

## Why the Output Head Must Always Be Trained

The pretrained GPT-2 output head maps:
```
768-dim hidden state → 50,257 vocabulary scores   (next-token prediction)
```

For binary classification (spam/ham), you need:
```
768-dim hidden state → 2 scores   (spam or not spam)
```

This is a completely **new layer** with random initial weights. It has never seen any data. Regardless of what else you freeze, this layer must always be trained — otherwise the classifier produces random outputs forever.

---

## Why Train the Final Transformer Block?

The final transformer block produces the hidden state that feeds directly into the output head. If you freeze it, the hidden state is optimized for next-token prediction — not for classification. Training just the final block allows the model to reshape its final representations to be useful for your specific task, without disturbing the lower blocks.

---

## Why Train the Final LayerNorm?

LayerNorm sits immediately before the output head. Its `scale` and `shift` parameters control the scale and center of the hidden states going into the classifier. If the pretrained values aren't well-calibrated for your new task, the classifier gets poorly-scaled inputs. Training the final LayerNorm lets it adapt this calibration cheaply (only 2 × 768 = 1,536 parameters).

---

## When to Unfreeze More Layers

This strategy works well for simple tasks on general-domain text. You might need to unfreeze more layers if:
- The task is complex or highly domain-specific (medical, legal, code)
- The dataset is large enough to avoid overfitting
- Initial results show underfitting (training loss doesn't decrease)

In that case, unfreeze one or two additional blocks from the top and monitor carefully.

---

## Related

- [[fine-tuning]]
- [[decoder-only-architecture]]
- [[layer-normalization]]
- [[gpt2-from-scratch]]
- [[spam-dataset-implementation]]
