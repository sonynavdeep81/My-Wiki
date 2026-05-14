---
title: Train vs Val vs Test Split — Why All Three?
type: query
tags: [training, evaluation, fine-tuning, classification]
sources: 1
updated: 2026-05-14
---

## Train vs Val vs Test Split — Why All Three?

**Summary**: The model trains only on train data. Val data guides human decisions (hyperparameter tuning, early stopping). Test data gives the final unbiased score — it is never used until the very end.

---

## The Three Splits

| Split | Model trains on it? | You make decisions based on it? | Purpose |
|---|---|---|---|
| Train | Yes — loss drives backprop | Yes | Weight updates, learning |
| Val | No — model never backprops here | Yes — tuning LR, epochs, architecture | Development-time feedback |
| Test | No | No — final eval only | Unbiased final score |

---

## Why Train Data Is Straightforward

The model directly optimizes its weights to minimize loss on training data. Every batch from training data flows through backpropagation and updates the weights. This is the definition of training.

---

## Why Val Data Is Needed

During training, you need to make decisions:
- Is the model overfitting? (val loss rising while train loss falls)
- Should I train for more epochs or stop now?
- Is this learning rate too high or too low?
- Is this architecture better than the last one I tried?

You cannot make these decisions using training data alone — training loss always decreases, even when the model is memorizing rather than learning. Val loss reveals whether the model is actually generalizing.

**The key point:** val data never directly updates the model's weights. The model sees val data only in forward pass mode — no gradients, no backpropagation.

---

## Why Val Data Is Not Truly "Unseen"

Although the model never trains on val data, it is not truly unseen either:

- You observe val loss → decide to stop training at epoch 5 instead of epoch 10
- You observe val accuracy → decide to unfreeze one more transformer block
- You observe val loss → choose learning rate 4e-4 over 1e-3

These decisions are informed by val data. Indirectly, val data influences the model you end up with. This is called **indirect information leakage** — the model selection process has been tuned using val data.

---

## Why Test Data Must Stay Completely Separate

Because val data leaks indirectly into model selection, it cannot give you a truly unbiased performance estimate. That is what test data is for.

Test data is locked away and never used until you have made all your decisions and settled on a final model. You evaluate on test data exactly once — to report your final number.

**If you ever look at test performance to make a decision, it is no longer a test set.** It has become another val set, and you need a new, truly unseen test set for an honest final evaluation.

---

## The Practical Rule

> If you have ever used a split to make a decision → it is not a test set anymore.

This is why benchmark leaderboards can be misleading — teams sometimes inadvertently tune their models based on test set feedback, which inflates their reported performance.

---

## Typical Split Ratios

For a dataset like SMS spam (~5,000 examples):

```
Train: 70–80%   → model learns from this
Val:   10–15%   → you watch this during training
Test:  10–15%   → you evaluate on this once, at the end
```

For very large datasets (millions of examples), even 1% for val/test gives enough samples for reliable estimates.

---

## Related

- [[fine-tuning]]
- [[training-loop-primitives]]
- [[classification-finetuning-strategy]]
