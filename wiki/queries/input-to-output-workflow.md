---
title: Complete Workflow: Input Text to Output Tokens
type: query
tags: [workflow, inference, tokenization, attention, decoding, end-to-end]
sources: 2
updated: 2026-04-17
---

## Complete Workflow: Input Text to Output Tokens

**Summary**: End-to-end shape trace from raw text → next token for a GPT-2 decoder-only LLM.

## Full Pipeline Diagram

```
INPUT TEXT
"Every effort takes you"
         │
         ▼
┌─────────────────────────────┐
│        TOKENIZATION         │
│  tiktoken BPE (GPT-2)       │
│  "Every"→464, "effort"→3797 │
│  "takes"→3332, "you"→319    │
└─────────────┬───────────────┘
              │  token IDs: [464, 3797, 3332, 319]   shape: (T,)
              ▼
┌─────────────────────────────────────────────────────┐
│                   EMBEDDING LAYER                    │
│                                                      │
│  Token IDs ──► tok_emb  (50,257 × 768)  ──► (T,768) │
│  [0,1,2,3]  ──► pos_emb (  256  × 768)  ──► (T,768) │
│                              +                       │
│                   final_input  (T, 768)              │
│                    + Dropout(0.1)                    │
└─────────────────────┬───────────────────────────────┘
                      │  (T, 768)
                      │
          ┌───────────┴───────────┐
          │   REPEAT ×12 BLOCKS   │
          │                       │
          │  ┌─────────────────┐  │
          │  │   LayerNorm     │  │
          │  └────────┬────────┘  │
          │           │           │
          │  ┌────────▼────────┐  │
          │  │ MASKED MULTI-   │  │
          │  │ HEAD ATTENTION  │  │
          │  │                 │  │
          │  │ X·W_Q → Q       │  │
          │  │ X·W_K → K  ×12  │  │
          │  │ X·W_V → V heads │  │
          │  │                 │  │
          │  │ Q·Kᵀ/√64        │  │
          │  │  + causal mask  │  │  ← future tokens → -∞
          │  │  → softmax      │  │
          │  │  → dropout      │  │
          │  │  ·V → concat    │  │
          │  │  ·W_O → (T,768) │  │
          │  └────────┬────────┘  │
          │           │           │
          │      + residual X     │
          │           │           │
          │  ┌────────▼────────┐  │
          │  │   LayerNorm     │  │
          │  └────────┬────────┘  │
          │           │           │
          │  ┌────────▼────────┐  │
          │  │  FEED-FORWARD   │  │
          │  │  768→3072(GELU) │  │
          │  │     →768        │  │
          │  └────────┬────────┘  │
          │           │           │
          │      + residual       │
          │                       │
          └───────────┬───────────┘
                      │  (T, 768)  ← rich contextual vectors
                      ▼
         ┌────────────────────────┐
         │     Final LayerNorm    │
         └────────────┬───────────┘
                      │
         ┌────────────▼───────────┐
         │   Linear Head          │
         │   768 → 50,257 logits  │
         └────────────┬───────────┘
                      │  (T, 50257) — only LAST ROW used at inference
                      ▼
         ┌────────────────────────┐
         │   DECODING STRATEGY    │
         │                        │
         │  1. top-k mask (k=25)  │  ← zero out all but top 25 logits
         │  2. ÷ temperature(1.4) │  ← sharpen/flatten distribution
         │  3. softmax → probs    │
         │  4. multinomial sample │
         └────────────┬───────────┘
                      │  next token ID, e.g. 2651 ("forward")
                      ▼
         ┌────────────────────────┐
         │   DETOKENIZE           │
         │   2651 → "forward"     │
         └────────────┬───────────┘
                      │
                      ▼
         Append to input → repeat until [EOS] or max_length

OUTPUT: "Every effort takes you forward ..."
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
