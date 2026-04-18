---
title: Why Concept Pages Exist
type: query
tags: [wiki, concept-pages, explanation]
updated: 2026-04-18
---

## Why Concept Pages Exist

**Summary**: Concept pages make the wiki faster, consistent, and token-efficient — not more accurate per se, but more scalable.

---

## What is a Concept Page?

When a source is ingested, important ideas are extracted and stored in small dedicated files called concept pages — one per topic. For example, `layer-normalization.md` stores everything the wiki knows about LayerNorm.

If two sources both talk about the same concept, both their perspectives end up in the same concept page:

```
Notebook says: Pre-LN   →  stored in layer-normalization.md
Paper says:    Post-LN  →  also stored in layer-normalization.md
```

This is topic-based storage instead of source-based storage.

---

## What is Synthesization?

Synthesization means combining information from multiple sources into one unified explanation — not summarizing each source separately.

Example: the notebook and the Attention paper both cover LayerNorm, but differently. The concept page captures both perspectives together, so comparisons and connections are already present in one place.

---

## The Three Real Benefits

### 1. Speed
Reading a 1MB notebook + a PDF paper at query time is slow. A concept page is 50–100 lines. If your question only needs LayerNorm facts, reading the concept page uses ~200 tokens vs 10,000+ for the full notebook.

### 2. Consistency
If sources are re-read fresh every time, different facts might be emphasized on different reads. Concept pages fix the key facts in one place — answers stay consistent across sessions.

### 3. Token savings
Fewer tokens read per query = lower cost and staying within context limits more easily. This matters more as the wiki grows.

---

## Does This Mean Raw Sources Are Useless?

No. Raw sources are ground truth — concept pages are built from them. If something needs verification, the raw source is checked. But for 90% of questions, the concept pages have everything needed.

---

## Would Accuracy Suffer Without Concept Pages?

Not necessarily for a small wiki. If sources are read carefully at query time, the same facts can be found. The accuracy benefit kicks in mainly at scale — when there are 10–20 sources, re-reading everything for every question becomes impractical, and that is when concept pages prevent real information loss.

---

## Simple Analogy

Raw sources = textbooks.
Concept pages = your personal notes that highlight key points and group them by topic across all textbooks.

Answering from notes is faster, cheaper, and more consistent than re-reading all the textbooks every time.

---

## Summary

The main benefits of concept pages are:
- **Faster** — small dense pages instead of full raw sources
- **Consistent** — facts fixed in one place, same answer every time
- **Token-efficient** — fewer tokens per query, scales as wiki grows
