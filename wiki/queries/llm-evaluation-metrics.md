---
title: Evaluation Metrics for a Decoder-Only LLM
type: query
tags: [evaluation, perplexity, loss, metrics, gpt2, training]
updated: 2026-05-14
---

## Evaluation Metrics for a Decoder-Only LLM

**Summary**: The primary metrics during pretraining are cross-entropy loss and perplexity. Generation quality is checked manually. Task-specific metrics (accuracy, F1) only apply after fine-tuning.

---

## Why Evaluation Is Different for LLMs

Unlike classification models where you can simply ask "did it predict the right label?", a decoder-only language model generates free-form text. There is no single correct next word — "the", "a", "my" could all be reasonable continuations of the same sentence. This makes evaluation more nuanced.

---

## Training-Time Metrics

### Cross-Entropy Loss

This is the direct training objective. At every token position, the model predicts a probability distribution over the 50,257 vocabulary words. Cross-entropy measures how surprised the model is by the actual next token.

```
loss = -1/T × Σ log P(actual_token | all_previous_tokens)
```

- Lower loss = model assigns higher probability to correct tokens = better
- Typical GPT-2 loss on validation data: around 3.0–3.5 (depends on dataset)
- The most important signal: is val_loss decreasing over training?

### Perplexity

Perplexity is cross-entropy loss converted to a more intuitive unit:

```
perplexity = exp(loss)
```

If perplexity = 29, it means on average the model is uncertain between about **29 equally likely choices** for the next token. Lower is better.

- GPT-2 124M achieves perplexity ~29 on WebText (its training data)
- On a different domain (e.g., medical text), perplexity will be much higher

### Train / Validation Gap

Comparing train loss to val loss tells you whether the model is overfitting:

```
gap = val_loss − train_loss
```

- Small gap → model generalizes well
- Widening gap → model is memorizing training data, not learning general patterns
- Val loss rising while train loss falls → clear sign of overfitting → stop training

---

## Generation Quality (Qualitative)

Numbers alone don't tell the whole story. You should also **manually read** a sample of generated text and check:

- **Coherence:** does the text make sense sentence by sentence?
- **Fluency:** does it read naturally, or is it choppy and awkward?
- **Repetition:** does the model get stuck in loops (e.g., "the the the the")?
- **Diversity:** does changing the sampling strategy (greedy vs top-k) produce meaningfully different outputs?

This qualitative check is especially important early in training when the model is still learning basic language patterns.

---

## Post Fine-Tuning Metrics

Once the model is fine-tuned for a specific task, task-specific metrics apply:

| Task | Metric |
|---|---|
| Binary classification (spam/ham) | Accuracy, F1 score |
| Instruction following | Win-rate vs baseline, human evaluation |
| Question answering | Exact match, F1 on answer spans |

---

## Metrics That Do NOT Apply to Pretraining

| Metric | Why it doesn't apply |
|---|---|
| BLEU | Designed for machine translation — measures n-gram overlap with a reference translation |
| ROUGE | Designed for summarization — measures recall of reference summary words |
| Exact Match | Requires a ground-truth answer — language modeling has no single correct output |

These metrics become relevant only after fine-tuning for specific tasks that have ground-truth reference outputs.

---

## Summary — What to Watch During Training

1. **Val loss curve** — the primary signal. Should decrease steadily, then plateau
2. **Train/val gap** — should stay small; widening means overfitting
3. **Perplexity** — easier to interpret than raw loss; should decrease over time
4. **Manual generation samples** — qualitative check for coherence and fluency

---

## Related

- [[gpt2-from-scratch]]
- [[decoding-strategies]]
- [[fine-tuning]]
- [[bleu-score]]
- [[perplexity]]
