---
title: Why Concept Pages Exist
type: query
tags: [wiki, concept-pages, explanation]
updated: 2026-05-14
---

## Why Concept Pages Exist

**Summary**: Concept pages make the wiki faster, consistent, and scalable — not more accurate per se, but far more practical as the number of sources grows.

---

## What Is a Concept Page?

When a source is ingested into the wiki, important ideas are extracted and stored in small dedicated files — one per topic. For example, `layer-normalization.md` stores everything the wiki knows about LayerNorm.

If two different sources both discuss the same concept, both perspectives end up in the same concept page:

```
Notebook says: GPT-2 uses Pre-LN (LayerNorm before attention)
Paper says:    Original Transformer uses Post-LN (LayerNorm after attention)
       ↓
Both stored together in layer-normalization.md
→ The comparison is already there, ready to use
```

This is **topic-based storage** instead of source-based storage. Instead of remembering "the notebook said X about LayerNorm", the wiki captures "here is everything known about LayerNorm, from all sources."

---

## What Is Synthesization?

Synthesization means combining information from multiple sources into one unified explanation — not summarizing each source separately.

Without synthesis: "The notebook says Pre-LN. The paper says Post-LN."

With synthesis: "Pre-LN places normalization before the sublayer (GPT-2 style); Post-LN places it after (original Transformer). Pre-LN training is more stable. The difference matters for gradient flow through deep networks."

Concept pages hold the synthesized view. Query time is spent answering your question, not re-reading and re-synthesizing sources.

---

## The Three Real Benefits

### 1. Speed

Reading a 1MB notebook and a PDF paper at query time is slow. A concept page is 50–100 lines. If your question only needs LayerNorm facts, reading the concept page uses ~200 tokens vs 10,000+ for the full notebook.

As the wiki grows from 5 sources to 20, this difference becomes critical. Re-reading everything for every question would eventually exceed the context window entirely.

### 2. Consistency

If sources are re-read fresh every time a question is asked, different facts might be emphasized on different reads. Concept pages fix the key facts in one place. The same question asked twice gets the same answer — not because the LLM remembers, but because it always reads from the same canonical source.

### 3. Token Efficiency

Fewer tokens read per query means lower cost and staying within context limits more easily. In a wiki with 20 sources totaling 2MB, reading everything for a simple question about LayerNorm would be wasteful. Concept pages let targeted lookup happen instead of full-corpus scanning.

---

## Are Raw Sources Useless Then?

No. Raw sources are the ground truth. Concept pages are built from them. If a fact needs verification or a conflict needs to be resolved, the raw source is checked. But for 90% of questions, the concept pages have everything needed.

Think of it like research notes:
- **Raw sources** = the original textbooks and papers
- **Concept pages** = your personal notes that highlight key points and group them by topic across all your reading

Answering from notes is faster, cheaper, and more consistent than re-reading all the textbooks every time — but the textbooks remain the authoritative source when accuracy is questioned.

---

## When Concept Pages Matter Most

For a small wiki (2–3 sources), concept pages add some overhead but not much benefit. You could re-read sources at query time without much penalty.

For a larger wiki (10–20 sources), concept pages become essential:
- Re-reading everything hits context limits
- Different sources say different things about the same topic — synthesis prevents confusion
- Consistency matters more as the number of facts grows

The wiki is designed to grow — concept pages are the infrastructure that makes growth sustainable.

---

## Related
