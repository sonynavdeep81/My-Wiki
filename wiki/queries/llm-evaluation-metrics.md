---
title: Evaluation Metrics for a Decoder-Only LLM
type: query
tags: [evaluation, perplexity, loss, metrics, gpt2, training]
updated: 2026-04-14
---

## Evaluation Metrics for a Decoder-Only LLM

**Summary**: For a decoder-only language model like your GPT-2 implementation, the primary metrics are cross-entropy loss and perplexity; generation quality is assessed manually; downstream task metrics apply only after fine-tuning.

---

## 1. Training-Time Metrics

### Cross-Entropy Loss
The direct training objective — average negative log-likelihood over all predicted tokens:

```
Loss = -1/T · Σ log P(token_t | token_1 ... token_{t-1})
```

- Lower = better
- Monitor both **train loss** and **validation loss**
- Widening gap between the two = overfitting

### Perplexity
Exponentiation of cross-entropy loss:

```
Perplexity = exp(Loss)
```

- Intuition: on average, how many tokens is the model equally confused between?
- Perplexity 10 → model is as uncertain as choosing uniformly among 10 tokens
- Perplexity 1 → perfect prediction
- GPT-2 (124M, OpenAI) achieves ~29 perplexity on WebText

**This is your primary metric** — computed directly from the loss already in your training loop.

---

## 2. Generation Quality (Qualitative)

Manual inspection of generated samples:

| Check | What to look for |
|---|---|
| Coherence | Does output make logical sense across sentences? |
| Fluency | Does it read naturally? |
| Repetition | Does it loop or degenerate? |
| Sampling sanity | Does greedy differ meaningfully from top-k? |

A model can have decent perplexity but still generate repetitive or incoherent text — manual checks catch this.

---

## 3. Downstream Task Metrics (Post Fine-Tuning)

Your GPT-2 notebook includes classification fine-tuning. Once fine-tuned, use task-specific metrics:

| Task | Metric |
|---|---|
| Text classification | Accuracy, F1 |
| Instruction following | Win-rate vs baseline, human eval |
| Text generation quality | Coherence scores, human preference |

See [[fine-tuning]] for fine-tuning approaches.

---

## What Does NOT Apply to Your Model

| Metric | Why not applicable |
|---|---|
| **[[bleu-score|BLEU]]** | For translation models — your model does not translate |
| **ROUGE** | For summarisation tasks |
| **Exact Match (EM)** | For QA with ground-truth answers |

---

## Summary for Your GPT-2

```
During training:    cross-entropy loss (train + val)
Primary LM metric:  perplexity = exp(val_loss)
Generation sanity:  manual inspection of sampled text
If fine-tuned:      accuracy / F1 on classification head
```

The validation loss curve is the most honest signal — if it plateaus or rises, the model has stopped improving or is overfitting.

---

## Related

- [[gpt2-from-scratch|GPT-2 From-Scratch Patterns]]
- [[decoding-strategies]]
- [[fine-tuning]]
- [[bleu-score]]
- [[large-language-models]]
