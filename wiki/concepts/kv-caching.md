---
title: KV Caching
type: concept
tags: [inference, optimization, attention, kv-cache]
sources: 1
updated: 2026-04-13
---

## KV Caching

**Summary**: An inference optimization that stores the Key and Value matrices of already-processed tokens so only the new token's attention needs to be computed at each generation step.

## The Problem

During [[decoder-only-architecture]] inference, each new token is appended to the input and the model runs a full forward pass. At every step, attention recomputes K and V for **all previous tokens** — even though they haven't changed. This is wasteful and slow.

## The Solution

**KV Caching** stores the K and V matrices (computed from W_K and W_V) for all previously processed tokens. At each new step:
- Retrieve cached K, V for previous tokens
- Compute K, V only for the **new token**
- Concatenate and compute attention

This avoids redundant matrix multiplications for the full context at every step.

## Trade-off

KV Cache reduces compute at the cost of **memory**. Storing K and V matrices for long contexts can be significant — this is a key engineering challenge for long-context models.

## Related

- [[multi-head-attention]]
- [[decoder-only-architecture]]
- [[inference-scaling]]
