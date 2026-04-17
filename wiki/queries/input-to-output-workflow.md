---
title: Complete Workflow: Input Text to Output Tokens
type: query
tags: [workflow, inference, tokenization, attention, decoding, end-to-end]
sources: 2
updated: 2026-04-17
---

## Complete Workflow: Input Text to Output Tokens

**Summary**: End-to-end shape trace from raw text → next token for a GPT-2 decoder-only LLM.

## Pipeline

```
"Every effort takes you"
  → tiktoken BPE → [464, 3797, 3332, 319]          shape: (T,)
  → tok_emb(50257×768) + pos_emb(256×768)           shape: (T, 768)
  → Dropout(0.1)
  → ×12 TransformerBlock:
      LayerNorm
      → MHA: Q=X·W_Q, K=X·W_K, V=X·W_V             (T,768)
             scores = Q·Kᵀ/√64                      (12,T,T)
             + causal mask (future→−∞)
             → softmax → Dropout → ·V → concat·W_O  (T,768)
      + residual
      LayerNorm
      → FFN: 768→3072(GELU)→768
      + residual
  → final LayerNorm
  → out_head Linear(768→50257)                      (T, 50257)
  → last row only                                   (50257,)
  → top-k(25) → ÷temperature(1.4) → softmax → multinomial
  → next token ID → detokenize → append → repeat
```

## Shape Trace

| Stage | Shape | Note |
|---|---|---|
| Token IDs | (T,) | integers |
| Embeddings | (T, 768) | tok + pos summed |
| Each block output | (T, 768) | unchanged shape |
| Logits | (T, 50257) | all positions |
| Inference logits | (50257,) | last row only |
| Sampled token | scalar | one new token |

## Key Facts

- Inference uses last row only; earlier rows used during training (teacher forcing)
- Forward pass re-runs on full sequence each step → why [[kv-caching]] matters
- Causal mask enables parallel training while preserving autoregressive order
- Decoding strategy applied only at final step; model forward is deterministic

## Related

- [[tokenization]]
- [[embeddings]]
- [[multi-head-attention]]
- [[decoder-only-architecture]]
- [[decoding-strategies]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[kv-caching]]
