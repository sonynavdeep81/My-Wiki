---
title: Dropout During Fine-Tuning — Why Set drop_rate=0.0
type: query
tags: [fine-tuning, dropout, regularization, classification, small-data]
sources: 0
updated: 2026-05-14
---

## Dropout During Fine-Tuning — Why Set drop_rate=0.0

**Summary**: When fine-tuning with most layers frozen on a small dataset, dropout hurts rather than helps. Set `drop_rate=0.0` so the small trainable head receives a clean, stable signal.

---

## What Is Dropout?

Dropout is a regularization technique. During training, it randomly zeros out a fraction of activations on every forward pass. For example, with `drop_rate=0.1`, 10% of the values in a layer are randomly set to zero each time.

The idea: by randomly silencing neurons, the model cannot rely on any single neuron too heavily. It is forced to learn redundant, distributed representations — which generalizes better.

---

## Why Dropout Helps During Pretraining

During pretraining on billions of tokens, the model sees each piece of data an enormous number of times:

```
effective looks ≈ num_tokens / context_length × num_epochs
               ≈ 10⁹ tokens → enormous repetition
```

With that much data, the random 10% silencing averages out statistically. Over millions of passes, every neuron gets to participate fully. The noise is just noise — it doesn't destroy the signal.

Dropout's benefit: prevents overfitting to specific patterns, forces robust feature learning. With 10⁹ effective looks, this benefit vastly outweighs the noise cost.

---

## Why Dropout Hurts During Partial Fine-Tuning

When fine-tuning a small classification dataset (e.g., ~1,000 spam/ham emails) for 10 epochs:

```
effective looks ≈ 1,045 examples × 10 epochs ≈ 10,000 total
```

With only ~10,000 looks, the random 10% silencing does **not** average out. The new output head (768 → 2, trained from scratch) sees flickering, inconsistent inputs — some neurons present, some randomly absent — on every pass.

The head cannot build stable feature detectors under these conditions. It's like trying to learn to recognize a face when 10% of the pixels are randomly blacked out each time — and you only have 10,000 chances total.

**Empirical evidence (SMS spam, GPT-2 124M, partial fine-tune):**

| Epoch | drop_rate=0.1 val_acc | drop_rate=0.0 val_acc |
|---|---|---|
| 3 | 67% | 75–80% |
| 4 | 82% | 85–88% |
| 5 | 82.5% | 85–88% |

Setting `drop_rate=0.0` gives ~5pp better accuracy and faster convergence.

---

## How Dropout Actually Works in PyTorch

`nn.Dropout` checks the model's training flag at every forward pass:

```python
model.train()   # training=True  → dropout fires (zeros drop_rate fraction)
model.eval()    # training=False → dropout is a no-op (all values pass through)
```

This means eval-time predictions are never affected by dropout regardless of `drop_rate`. The `drop_rate` value only matters during training.

---

## The Fix — Set drop_rate=0.0 in Config

```python
GPT_CONFIG_124M['drop_rate'] = 0.0   # disables all dropout layers globally
model = GPT2Model(GPT_CONFIG_124M)
```

Every `nn.Dropout(p=0.0)` instance becomes a mathematical identity — it passes all values through unchanged. You get clean signal during training without any other changes to your code.

---

## Why Not Just Use model.eval() During Training?

You might think: "if `model.eval()` disables dropout, why not train under eval mode?"

Technically it would work for GPT-2 (which has no BatchNorm). But it is non-idiomatic and dangerous:

| | model.train() | model.eval() |
|---|---|---|
| Dropout | active | off |
| BatchNorm | uses batch stats | uses running stats |
| Convention | training loop | val/test only |
| Future compatibility | safe | fragile |

Any future code that branches on `model.training` (a common pattern) would break silently. The clean solution is to keep `model.train()` in the training loop and `model.eval()` for validation — and simply set `drop_rate=0.0` so dropout is a no-op everywhere.

**Important:** `drop_rate` is baked into `nn.Dropout(p=...)` at construction time. You cannot change it after the model is built without rebuilding the model or iterating through all modules. Always set it in the config before creating the model.

---

## Decision Table

| Scenario | drop_rate | Reason |
|---|---|---|
| Pretraining (~10⁹ tokens, all params) | 0.1 | Noise averages out; strong regularization benefit |
| Full fine-tune (all blocks, small data) | 0.05–0.1 | Many trainable params; overfit risk |
| Partial fine-tune (mostly frozen + tiny head) | **0.0** | Head's signal too small; noise doesn't average |

---

## When to Bring Dropout Back

- You unfreeze many transformer blocks (overfit risk returns)
- Your fine-tuning dataset is large (>10,000 examples)
- You observe overfitting: training loss falls but val loss rises

---

## Related

- [[dropout]]
- [[fine-tuning]]
- [[classification-finetuning-strategy]]
- [[gpt2-from-scratch]]
- [[training-loop-primitives]]
