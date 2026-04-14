---
title: Decoder-Only Architecture
type: concept
tags: [decoder, transformer, gpt, causal-masking, training, inference]
sources: 1
updated: 2026-04-13
---

## Decoder-Only Architecture

**Summary**: A transformer variant that uses only the decoder stack with causal masking, enabling autoregressive text generation; used by GPT, LLaMA, Mistral, and most modern LLMs.

## Overview

The decoder-only model stacks Nx identical transformer blocks (Nx=12 for GPT-2) on top of token + positional embeddings. Each block contains:
1. **Pre-LN** (layer norm before sublayer)
2. **Masked Multi-Head Attention** (causal)
3. Residual connection
4. Pre-LN
5. **FFN** (768→3072→768)
6. Residual connection

Final output: Linear projection → Softmax → vocabulary probability distribution.

## Training: Teacher Forcing

Input: "Every effort takes you"

| Token | Context Seen | Predicts |
|---|---|---|
| "Every" | "Every" only | "effort" |
| "effort" | "Every, effort" | "takes" |
| "takes" | "Every, effort, takes" | "you" |
| "you" | "Every, effort, takes, you" | "forward" |

All 4 positions are computed **in parallel** (thanks to causal masking). During training, the model always receives the actual correct tokens as context — this is **teacher forcing**. Loss is averaged across all 4 predictions; gradients updated together.

## Inference: Sequential

After training, inference is sequential. Given "Every effort takes you":
- Model outputs 4 rows of logits; we only care about the last row ("you" → predicts "forward")
- "forward" is appended → new input "Every effort takes you forward" → predict next → repeat until `[EOS]` or max length

Inference is inherently slow because each step appends one token. **[[kv-caching]]** mitigates this by storing K/V matrices from previous steps.

## Exposure Bias

Training uses ground-truth tokens (teacher forcing); inference uses the model's own predictions. If the model makes a wrong prediction, that wrong token is fed into the next step, potentially cascading errors. This gap is called **exposure bias**.

## Context Length

Only the last `context_length` tokens are used. The attention matrix grows as (context_length × context_length), making longer contexts quadratically more expensive. This is why context length is a critical design constraint.

## Pre-LN vs Post-LN

Modern decoder-only models use **Pre-Layer Normalization** (normalize before the sublayer). This is more stable and trains faster than the Post-LN in the original "Attention is All You Need" paper. See [[layer-normalization]].

## Related

- [[transformer-architecture]]
- [[multi-head-attention]]
- [[kv-caching]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[embeddings]]
