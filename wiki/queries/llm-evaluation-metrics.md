---
title: Evaluation Metrics for a Decoder-Only LLM
type: query
tags: [evaluation, perplexity, loss, metrics, gpt2, training]
updated: 2026-04-17
---

## Evaluation Metrics for a Decoder-Only LLM

**Summary**: Primary metrics are cross-entropy loss + perplexity; generation quality checked manually; task metrics only after fine-tuning.

## Training-Time Metrics

| Metric | Formula | Note |
|---|---|---|
| Cross-entropy loss | `-1/T · Σ log P(tₜ \| t<ₜ)` | Direct training objective; lower=better |
| Perplexity | `exp(loss)` | Avg tokens model is uncertain between; GPT-2 124M ~29 on WebText |
| Train/val gap | val_loss − train_loss | Widening → overfitting |

**Primary signal**: val_loss curve. Plateau or rise → stopped improving or overfitting.

## Generation Quality (Qualitative)

Manual inspection: coherence, fluency, repetition/degeneration, sampling diversity (greedy vs top-k).

## Post Fine-Tuning Metrics

| Task | Metric |
|---|---|
| Classification | Accuracy, F1 |
| Instruction following | Win-rate, human eval |

See [[fine-tuning]].

## Not Applicable

| Metric | Reason |
|---|---|
| [[bleu-score\|BLEU]] | Translation only |
| ROUGE | Summarisation only |
| Exact Match | QA with ground-truth answers |

## Related

- [[gpt2-from-scratch]]
- [[decoding-strategies]]
- [[fine-tuning]]
- [[bleu-score]]
- [[large-language-models]]
