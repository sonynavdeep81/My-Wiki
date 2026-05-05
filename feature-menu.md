---
title: Wiki Feature Menu
description: Features developed in the Agents wiki — pick what suits your wiki's purpose
updated: 2026-05-05
---

# Wiki Feature Menu

Features developed in the Agents learning wiki. Each is independent — pick only what fits your wiki's purpose and rewrite the instructions in your own CLAUDE.md in your own terms.

---

## 1. Concept Pages as Source of Truth

**What it does:** When a Q&A session deepens understanding of a topic, that depth is written directly into the concept page — not only saved in the query page. Concept pages are always the primary knowledge store.

**Rule to add:** After every answer, if depth was added that the concept page doesn't have, update the concept page immediately.

**Best for:** Any wiki where you want knowledge to accumulate over time, not stay buried in saved conversations.

**Not needed if:** Your wiki is purely reference-based with no Q&A workflow.

---

## 2. Query Page Types — Breadcrumb vs Format Artifact

**What it does:** Distinguishes two kinds of saved Q&A pages:
- **Breadcrumb** — short summary + links to concept pages. Used when the answer draws from existing pages.
- **Format artifact** — full content kept. Used when the output format itself is the value (slide deck, study notes, audience-specific explainer).

**Rule to add:** Default to breadcrumb. Only use format artifact when the format is the point.

**Best for:** Any wiki that saves Q&A outputs and wants to avoid duplication.

**Not needed if:** You never save Q&A as wiki pages.

---

## 3. Gap Tracking (`wiki/gaps/`)

**What it does:** On every ingest, topics that are mentioned but not deeply covered are logged immediately as gap files. Gaps accumulate continuously rather than being discovered only at lint time.

**Rule to add:** On every ingest, for each topic the source mentions but doesn't explain deeply, create or update a gap file. If a gap is filled by a new ingest, remove or close it.

**Best for:** Research wikis and learning wikis that ingest multiple sources over time.

**Not needed if:** Your wiki has a fixed, complete set of sources with no ongoing ingestion.

---

## 4. Learning Path with Gap Visibility

**What it does:** A `learning-path.md` file defines the pedagogical sequence for all concepts — foundational first, patterns second, scaling third, failure modes last. It also includes a `## Gaps in This Path` section listing topics that belong in the sequence but have no page yet.

**Rule to add:** On every ingest, update the learning path — insert new concept pages in the right stage, remove filled gaps, add new ones.

**Best for:** Learning wikis where you want a clear reading order and visibility into what's still missing.

**Not needed if:** Your wiki is a reference/research wiki with no intended reading sequence.

---

## 5. Slide Deck Follows Learning Path

**What it does:** When asked for a slide deck, Claude follows the learning path stage order exactly — one slide group per stage, skipping any stage that's in the gaps section.

**Rule to add:** Slide decks follow learning-path.md stage order. Skip gap stages silently.

**Best for:** Learning wikis where you present content to others (students, colleagues).

**Not needed if:** You don't make slides, or your wiki isn't structured as a learning sequence.

---

## 6. Proactive Contradiction Checking on Ingest

**What it does:** Before writing any files during an ingest, new claims are compared against existing wiki pages. Conflicts are surfaced to the user with severity rating and a reasoned position — never silently resolved.

**Rule to add:** On ingest, check new claims against existing pages. Report conflicts before writing. Always take a position on which claim is more likely correct.

**Best for:** Research wikis and any wiki ingesting multiple sources that may disagree.

**Not needed if:** Your wiki has a single authoritative source.

---

## 7. Source Transparency on Answers

**What it does:** If any part of an answer draws on general training knowledge rather than the wiki, it's explicitly flagged: *"Note: this is from my general knowledge, not the wiki."*

**Rule to add:** Always flag answers that go beyond wiki content.

**Best for:** Any wiki where you want to know if your knowledge base actually covers a topic or if Claude is filling in from training data.

**Not needed if:** You're comfortable with Claude blending wiki and general knowledge freely.

---

## 8. Lint as Structural Health Check (not knowledge discovery)

**What it does:** Lint focuses on structure — orphan pages, broken wikilinks, contradiction review, gap status check. Knowledge discovery is handled by gap tracking (feature 3), not lint.

**Rule to add:** Lint = structural check only. It does not need to rediscover gaps if gap tracking is active.

**Best for:** Any wiki that uses gap tracking. Keeps lint fast and focused.

**Not needed if:** You're not using gap tracking — in that case, lint should still include knowledge gap discovery.

---

## Combinations That Work Well Together

| Wiki Type | Recommended Features |
|-----------|---------------------|
| Learning wiki | 1, 2, 3, 4, 5, 7, 8 |
| Research wiki | 1, 2, 3, 6, 7, 8 |
| Reference wiki (fixed sources) | 1, 2, 6, 7 |
| Presentation / notes wiki | 2, 4, 5, 7 |
