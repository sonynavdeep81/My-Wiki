---
title: Emergent Abilities
type: concept
tags: [emergence, scaling, capabilities, llm]
sources: 1
updated: 2026-04-13
---

## Emergent Abilities

**Summary**: Capabilities that arise spontaneously in LLMs — like reasoning, math, and coding — that were never explicitly trained, emerging as a side effect of next-token prediction at sufficient scale.

## What They Are

LLMs are only trained to predict the next token. Yet they spontaneously learn:
- Grammar and syntax
- Translation (across languages)
- Mathematical reasoning
- Logic and coding
- World knowledge

These were not explicitly trained — they **emerge naturally** during pretraining.

## Why They Happen

Next-token prediction forces the model to:
- Understand syntax (to form grammatically correct sentences)
- Grasp meaning and context (to stay coherent)
- Track facts and logic (to avoid contradictions)
- Learn cross-lingual structure (which leads to translation ability)

## Connection to Scaling

Emergent abilities are tied to [[scaling-laws]]. Once a model reaches a critical threshold of parameters × data × compute, performance on certain tasks **jumps suddenly from near-zero to significant levels** — not gradually. This makes emergence hard to predict in advance.

## Related

- [[scaling-laws]]
- [[large-language-models]]
- [[inference-scaling]]
