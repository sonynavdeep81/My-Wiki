---
title: Complete Workflow: Input Text to Output Tokens
type: query
tags: [workflow, inference, tokenization, attention, decoding, end-to-end]
sources: 2
updated: 2026-04-14
---

## Complete Workflow: Input Text to Output Tokens

**Summary**: End-to-end walkthrough of how a decoder-only LLM (GPT-2) transforms raw input text into output tokens, from tokenization through embedding, 12 transformer blocks, logit projection, and decoding.

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

| Stage | Shape | What changes |
|---|---|---|
| Raw text | string | — |
| Token IDs | (4,) | Text → integers |
| Embeddings | (4, 768) | Integers → dense vectors |
| After each block | (4, 768) | Vectors become more contextual |
| After 12 blocks | (4, 768) | Full context captured |
| Logits | (4, 50257) | Project to vocab space |
| Probabilities | (50257,) | Last row only; softmax applied |
| Next token | scalar | One new token sampled |

## Key Points

- **Only the last row of logits is used at inference** — earlier rows were useful during training (teacher forcing) but are discarded at inference time
- **The loop is sequential**: each new token is appended and the full forward pass re-runs — this is why inference is slow and why [[kv-caching]] matters
- **Transformer blocks are identical in structure** but each has its own independent weights — early blocks learn syntax/grammar, middle blocks learn semantics, late blocks learn reasoning/world knowledge
- **Causal mask** ensures token at position i can only attend to positions ≤ i — this is what allows parallel training while preserving autoregressive structure
- **Decoding strategy** (top-k + temperature) is applied only at the final step — the model itself is deterministic up to that point

## Related

- [[tokenization]]
- [[embeddings]]
- [[multi-head-attention]]
- [[decoder-only-architecture]]
- [[decoding-strategies]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[kv-caching]]
