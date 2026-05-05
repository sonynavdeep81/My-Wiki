---
title: Dropout During Fine-Tuning — Why Set drop_rate=0.0
type: query
tags: [fine-tuning, dropout, regularization, classification, small-data]
sources: 0
updated: 2026-04-26
---

## Dropout During Fine-Tuning — Why Set drop_rate=0.0

**Summary**: When fine-tuning with most layers frozen on a small dataset, set drop_rate=0.0; dropout's noise no longer averages out and slows the small trainable head's convergence.

## Decision Table

| Scenario | drop_rate | Reason |
|---|---|---|
| Pretraining (~10⁹ tokens, all params trainable) | 0.1 | Random silencing forces feature-sharing across neurons; noise averages over millions of repetitions |
| Full fine-tune (all blocks unfrozen, small data) | 0.05–0.1 | Need regularization to prevent overfit; many trainable params can absorb noise |
| **Partial fine-tune (mostly frozen + tiny head)** | **0.0** | New head's signal already small; dropout noise doesn't average on ~10⁴ total looks |

## Why It Hurts in Partial Fine-Tune

- Effective looks at data = num_examples × num_epochs ≈ 1045 × 10 = ~10⁴
- Pretraining: ~10⁹ effective looks → random 10% silencing statistically averages to clean signal
- Fine-tune: ~10⁴ effective looks → noise does NOT average; head sees flickering input
- New `out_head` (768→2) trained from scratch; needs clean upstream signal to extract spam vs ham
- Train (10% silent) ≠ eval (0% silent) → mild calibration mismatch

## Mechanism

- Dropout is a **train-mode-only** operation; `nn.Dropout` checks `self.training` flag
- `model.train()` → `training=True` → dropout fires (zeros 10% of activations per forward pass)
- `model.eval()` → `training=False` → dropout is a no-op (passes activations through unchanged)
- Eval-time predictions never see dropout regardless of `drop_rate` value

```
forward (train mode):
  drop_rate=0.1 → 10% zeroed per pass (different mask each call)
  drop_rate=0.0 → no-op (deterministic; clean signal forward)

forward (eval mode): dropout always no-op (drop_rate ignored)
```

## Why drop_rate=0.0 Instead of Just Using model.eval()

Fine-tuning **requires** training mode for the conventional workflow:

| | model.train() | model.eval() |
|---|---|---|
| Dropout | **active** | off |
| Convention | training loop | val/test eval |
| Gradient flow | yes | yes (still works) |
| Optimizer step | yes | yes (still works) |

Technically you *could* train under `model.eval()` (gradients still flow, weights still update — only dropout/BatchNorm flag changes). For GPT-2 with no BatchNorm, this would even work. But it's non-idiomatic and breaks any future code that branches on `model.training`.

The clean solution: keep the standard `model.train()` during the training loop and `model.eval()` during val/test, but make dropout a **no-op everywhere** by setting `drop_rate=0.0` at model construction:

```python
GPT_CONFIG_124M['drop_rate'] = 0.0   # disables all dropout layers globally
model = GPT2Model(GPT_CONFIG_124M)
```

Now `model.train()` enables training mode but every `nn.Dropout` instance has `p=0` and acts as identity. You get clean signal during training without abusing eval mode.

Note: `drop_rate` is baked into `nn.Dropout(p=...)` at construction. Changing it after init requires either rebuilding the model or iterating modules and setting `m.p = 0.0` for each `nn.Dropout` — cleaner to set it in cfg before construction.

## Empirical (SMS spam, GPT-2 124M, freeze all except trf_blocks[-1] + final_norm + new out_head)

| epoch | drop_rate=0.1 val_acc | expected drop_rate=0.0 |
|---|---|---|
| 3 | 67% | 75–80% |
| 4 | 82% | 85–88% |
| 5 | 82.5% | 85–88% (plateau) |

Estimated lift: +1–2pp on val_acc; faster convergence (epoch 3–4 vs 5).

## When to Bring Dropout Back

- Unfreezing many blocks → overfit risk returns
- Larger fine-tune dataset (>10⁴ examples)
- Trainable params >> head-only setup

## Beginner Analogy

Studying photos of a friend's face with 10% randomly blacked out:
- 1M photos → fine, every part shows up plenty
- 1000 photos × 10 looks → details get blacked out 3× in a row, you finish with gaps

Pretraining = 1M photos. Fine-tune = 1000 photos.

## Related

- [[dropout]]
- [[fine-tuning]]
- [[classification-finetuning-strategy]]
- [[gpt2-from-scratch]]
- [[training-loop-primitives]]
