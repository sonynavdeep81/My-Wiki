---
title: Decoding Strategies (Temperature, Top-k, Sampling)
type: concept
tags: [inference, sampling, temperature, top-k, generation, decoding]
sources: 1
updated: 2026-04-13
---

## Decoding Strategies (Temperature, Top-k, Sampling)

**Summary**: Methods for converting model logits into the next token during inference, balancing creativity vs. coherence via temperature scaling and top-k filtering.

## The Problem

After a forward pass, the model outputs raw **logits** (shape: `(batch, vocab_size)`). We need a strategy to pick the next token from 50,257 candidates.

## Greedy Decoding

Always pick `argmax(logits)`. Deterministic and fast but repetitive — the model gets stuck in loops.

## Temperature Scaling

Divide logits by a temperature scalar **before** softmax:

```python
probs = torch.softmax(logits / temperature, dim=-1)
```

| Temperature | Effect |
|---|---|
| `= 1.0` | Probabilities unchanged |
| `< 1.0` | High probs → higher; low probs → lower (sharper, more confident) |
| `> 1.0` | High probs → lower; low probs → higher (flatter, more random) |
| `→ 0` | Fully greedy (argmax always wins) |
| `→ ∞` | Fully uniform (all tokens equally likely) |

Notebook default: `temperature = 1.4` (creative, slightly wild).

## Top-k Sampling

**Problem with temperature alone**: even very low-probability tokens can be sampled, producing nonsensical output.

**Solution**: restrict the candidate pool to the top-k most likely tokens before softmax.

```python
top_values, _ = torch.topk(logits, k=top_k)
min_val = top_values[:, -1]              # k-th highest value
logits[logits < min_val] = float('-inf') # mask everything below
probs = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

After masking, softmax converts `−∞` → `0`, so probability mass redistributes only over the top-k tokens.

Notebook default: `top_k = 25`.

## Why Top-k and Temperature Work Together

- **Top-k** controls *which* tokens are candidates (quality floor)
- **Temperature** controls *how confidently* to pick among them (creativity dial)

Using only top-k → can still be deterministic if k=1 (greedy). Using only temperature → still samples from full vocab with tiny noise. Together they cover the quality-creativity trade-off.

## torch.multinomial()

Samples one (or more) indices proportionally to the given probability tensor:
- Token A: 0.6 → picked 60% of the time
- Token B: 0.3 → picked 30% of the time
- Token C: 0.1 → picked 10% of the time

```python
next_token = torch.multinomial(probs, num_samples=1)
```

## Batched Generation

GPT-2 has no padding concept by design. For batched inference with inputs of different lengths, pad shorter sequences with `tokenizer.eot_token` (`<|endoftext|>`). The causal mask ensures padding tokens in earlier positions don't affect the generated tokens.

## Full Generation Loop

```python
model.eval()
with torch.no_grad():
    for _ in range(max_length):
        logits = model(token_ids)[:, -1, :]          # (batch, vocab)
        top_values, _ = torch.topk(logits, k=top_k)
        logits[logits < top_values[:, -1:]] = -inf
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token], dim=-1)
```

## Related

- [[decoder-only-architecture]]
- [[gpt2-from-scratch]]
- [[kv-caching]]
