---
title: BLEU Score
type: concept
tags: [evaluation, bleu, machine-translation, nlp]
sources: 1
updated: 2026-04-14
---

## BLEU Score

**Summary**: Bilingual Evaluation Understudy — an automatic metric for machine translation quality that measures n-gram overlap between a model's output and human reference translations.

---

## How It Works

BLEU counts how many n-grams (1-gram through 4-gram) in the model's output appear in the reference translation, with a brevity penalty for outputs that are too short:

```
BLEU = BP · exp( Σ wₙ · log pₙ )

where:
  pₙ  = precision of n-grams (n = 1..4)
  wₙ  = weight (typically 1/4 each)
  BP  = brevity penalty (penalises short outputs)
```

Score ranges from **0 to 100**. Higher = closer to human translation.

---

## Interpreting Scores

| BLEU | Interpretation |
|---|---|
| < 10 | Almost useless |
| 10–19 | Hard to understand |
| 20–29 | Clear in parts |
| 30–40 | Understandable and fluent |
| 40–50 | High quality |
| 50+ | Near human quality |

---

## Results in the Attention Is All You Need Paper

The paper used BLEU as the primary metric on WMT 2014:

| Task | Model | BLEU |
|---|---|---|
| English → German | Transformer (big) | **28.4** |
| English → French | Transformer (big) | **41.8** |

Both results were SOTA at the time, achieved at a fraction of prior training cost.

**Not applicable to your GPT-2 implementation** — GPT-2 is a language model evaluated on perplexity, not a translation model evaluated on BLEU.

---

## Limitations

- Does not capture meaning — a grammatically valid but semantically wrong sentence can score well
- Sensitive to tokenization choices
- Poor correlation with human judgement on short outputs
- Largely superseded by learned metrics (BERTScore, BLEURT, COMET) in research, but still widely reported for comparability

---

## Related

- [[Attention_2023|Attention Is All You Need (Paper)]]
- [[attention-is-all-you-need]]
- [[large-language-models]]
- [[tokenization]]
