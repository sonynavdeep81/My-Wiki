---
title: Train vs Val vs Test Split — Why All Three?
type: query
tags: [training, evaluation, fine-tuning, classification]
sources: 1
updated: 2026-04-24
---

## Train vs Val vs Test Split — Why All Three?

**Summary**: Model trains only on train data; val guides human decisions; test gives final unbiased score.

| Split | Model sees weights updated? | You make decisions from it? |
|-------|----------------------------|-----------------------------|
| Train | Yes | Yes (loss drives backprop) |
| Val | No | Yes (tune LR, epochs, arch) |
| Test | No | No (final eval only) |

**Why val data isn't "unseen" in the pure sense:**
- Model never trains on val → no direct weight update
- But you observe val loss and adjust hyperparameters, early stopping, architecture
- Those decisions are indirectly influenced by val data → val data leaks into model selection

**Why test data is kept separate:**
- You never look at test loss until the very end
- No decisions made based on it → truly unbiased estimate of generalization

**Practical rule:** If you've ever used a split to make a decision → it's not a test set anymore.

## Related

- [[Fine-Tuning]]
- [[Training Loop Primitives]]
- [[Classification Fine-Tuning Strategy — What to Freeze and What to Train]]
