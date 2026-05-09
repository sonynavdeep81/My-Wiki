---
title: BERT (Bidirectional Encoder Representations from Transformers)
type: concept
tags: [bert, encoder, transformer, classification, masked-prediction, nlp]
sources: 1
updated: 2026-05-07
verified_against: Raschka-LLM-2025, 2026-05-07
confidence: high
---

## BERT

**Summary**: Encoder-only transformer trained via masked token prediction; bidirectional context gives strong classification performance but weak generative ability.

## Architecture

- Uses only the **encoder** half of the original [[transformer-architecture]]
- Bidirectional: each token attends to ALL other tokens (no causal mask)
- Training objective: predict randomly masked tokens from full context ("fill in the blank")

```
Input:  "This is an __ of how concise I __ be"
Output: "example" and "can"
```

## BERT vs GPT

| Property | BERT | GPT |
|---|---|---|
| Transformer part | Encoder only | Decoder only |
| Attention | Bidirectional | Causal (left-to-right) |
| Training objective | Masked token prediction | Next-token prediction |
| Zero/few-shot | Weak | Strong |
| Classification | Strong natively | Via fine-tuning |
| Text generation | Cannot | Can |
| Example use | Sentiment analysis, NER | Chat, completion, instruction following |

## Use Cases

- Text classification (sentiment, spam, topic)
- Named entity recognition
- Question answering (extractive)
- X (Twitter) uses BERT to detect toxic content (Raschka Ch. 1)

## Variants

- **RoBERTa** (Liu et al. 2019): robustly optimized BERT pretraining
- **DistilBERT**: compressed/distilled BERT

## Recent Work

Classification performance from GPT-style models can match BERT by removing the causal mask during fine-tuning (Li et al. 2023 "Label Supervised LLaMA Finetuning"; BehnamGhader et al. 2024 "LLM2Vec"). [single-source]

## Related

- [[transformer-architecture]]
- [[decoder-only-architecture]]
- [[zero-shot-few-shot]]
- [[fine-tuning]]
- [[causal-masking]]
