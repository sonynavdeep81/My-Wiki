---
title: Cross-Entropy Loss
type: concept
tags: [loss, training, cross-entropy, ignore_index, softmax]
sources: 2
updated: 2026-05-05
verified_against: classification_fine_tuning, 2026-05-05
confidence: high
---

## Cross-Entropy Loss

**Summary**: Standard loss function for next-token prediction and classification; measures how well the model's predicted probability distribution matches the true label.

## Formula

```
CE = -log(p_correct)
```

- `p_correct` = softmax probability assigned to the true token
- High confidence on correct token → CE near 0
- Low confidence or wrong token → CE large
- Averaged over all (non-masked) positions in the batch

## PyTorch API

```python
# Language modeling (next-token prediction)
loss = torch.nn.functional.cross_entropy(logits, targets)
# logits: (batch, seq_len, vocab_size)  or  (batch*seq_len, vocab_size)
# targets: (batch, seq_len)             or  (batch*seq_len,)

# Classification (single label per sample)
loss = torch.nn.functional.cross_entropy(logits, targets)
# logits: (batch, num_classes)
# targets: (batch,)  — integer class indices
```

## ignore_index=-100

`cross_entropy` accepts `ignore_index` (default -100). Any target position set to -100:
- Excluded from loss computation entirely (not zeroed — fully skipped)
- Excluded from gradient computation
- Used in instruction fine-tuning to mask instruction tokens and padding tokens

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
```

## Relationship to Perplexity

```
Perplexity = exp(cross_entropy_loss)
```

If `loss = 3.0` → perplexity ≈ 20 (model is as confused as choosing uniformly over 20 tokens).

## Relationship to Softmax

`cross_entropy` in PyTorch combines `log_softmax + NLLLoss` internally — do **not** apply softmax to logits before passing to `cross_entropy` (double-softmax error).

## Source Differences

| | GPT-2 (your code) | Attention Is All You Need |
|---|---|---|
| Loss | `cross_entropy` (no smoothing) | `cross_entropy` + label smoothing ε=0.1 |
| ignore_index | -100 (instruction FT) | not used |

## Related

- [[perplexity]]
- [[softmax]]
- [[training-loop-primitives]]
- [[instruction-finetuning-training-mechanics]]
- [[instruction-fine-tuning]]
- [[label-smoothing]]
