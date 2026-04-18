---
title: Perplexity
type: concept
tags: [evaluation, metrics, language-model, loss, perplexity]
sources: 1
updated: 2026-04-18
---

## Perplexity

**Summary**: Standard intrinsic evaluation metric for language models; exponentiated cross-entropy loss — lower is better.

## Formula

```
PPL = exp(H)  where H = -(1/N) Σ log P(x_i | x_{<i})
```

Equivalently:
```python
PPL = torch.exp(loss)   # where loss = cross_entropy
```

- H = average negative log-likelihood per token (= cross-entropy loss)
- PPL = average branching factor the model "sees" at each step
- PPL = 1 → perfect prediction; PPL = vocab_size → random guessing

## Intuition

PPL of 50 means: on average, the model is as confused as if choosing uniformly from 50 equally likely tokens at each step.

## Relationship to Loss

| Cross-entropy loss | Perplexity |
|-------------------|------------|
| 0.0 | 1.0 (perfect) |
| 1.0 | 2.72 |
| 3.0 | 20.1 |
| 4.6 | 100 |
| ln(50257) ≈ 10.8 | 50257 (random) |

## Limitations

- Sensitive to tokenization — PPL values only comparable within the same tokenizer
- Does not measure usefulness, factuality, or creativity
- Rewards safe/repetitive predictions over diverse/creative ones

## In Practice (GPT-2 notebook)

```python
val_loss = cal_loader_loss(val_loader, model)
ppl = torch.exp(torch.tensor(val_loss))
```

Not explicitly computed in the notebook — but val_loss is the direct input.

## Related

- [[large-language-models]]
- [[optimizer]]
- [[gpt2-from-scratch]]
- [[scaling-laws]]
