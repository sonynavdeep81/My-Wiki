# LLM Wiki — Schema & Rules

## Purpose

This is a personal research wiki on "LLM internals and NLP".
The LLM writes and maintains all files in wiki/. I rarely edit them directly.

## Directory Structure

- raw/ → immutable source documents. Never modify these.
- wiki/concepts/ → concept articles (one per key idea)
- wiki/sources/ → summary per raw source
- wiki/entities/ → people, tools, papers, models
- wiki/queries/ → saved Q&A outputs
- wiki/gaps/ → one file per knowledge gap; created on ingest, removed when filled
- wiki/experiments/ → one file per experiment run (for active research papers)
- wiki/lint/ → lint run outputs (structural checks only)
- index.md → master catalog with one-line summary per page
- log.md → append-only log, format: ## [YYYY-MM-DD] action | title
- learning-path.md → canonical reading order across all concept pages

## On Ingest (when I say "ingest raw/filename.md"):

1. Read the source carefully
2. **Before writing anything:** compare new claims against existing concept pages. Report any conflicts to the user with severity (minor / significant) and a reasoned position on which claim is more likely correct. Never silently resolve conflicts.
3. Discuss key takeaways with me briefly
4. Create wiki/sources/[filename].md with summary + key points + [[backlinks]]
5. Create or UPDATE wiki/concepts/\*.md for each concept found
6. Create or UPDATE wiki/entities/\*.md for tools/people mentioned
7. **Gap tracking:** for each topic the source mentions but doesn't explain deeply, create or update wiki/gaps/[topic].md. If a new ingest fills an existing gap, delete that gap file.
8. Update learning-path.md — insert new concept pages in the correct stage; remove newly filled gaps; add new gap placeholders
9. Update index.md — add new pages, update summaries
10. Append to log.md

## On Query (when I ask a question):

1. Read index.md to find relevant pages
2. Read those pages fully
3. Synthesize answer with [[wikilink]] citations
4. **Source transparency:** if any part of the answer draws on general training knowledge rather than the wiki, flag it explicitly: *"Note: this is from my general knowledge, not the wiki."*
5. **Concept page update:** if the answer added depth not already in the relevant concept pages, update those pages immediately — do not wait for an ingest.
6. Ask me: "Should I file this answer as a wiki page?"
7. If yes, choose the type:
   - **Breadcrumb** — short summary + links to concept pages. Use when the answer draws from existing pages and the depth now lives there.
   - **Format artifact** — full content kept. Use when the output format itself is the value (study notes, audience-specific explainer, structured comparison). Default to breadcrumb.

## On Lint (when I say "lint the wiki"):

Lint is a **structural health check only** — knowledge gap discovery is handled continuously by gap tracking on ingest.

- Find orphan pages (no inbound links)
- Find broken wikilinks (references to pages that don't exist)
- Review open contradiction flags — check if any were resolved by recent ingests
- Check gap files — list which are still open, which are stale
- Check learning-path.md — verify all concept pages are placed; flag any missing
- Append findings to wiki/lint/lint-[date].md

## Format for wiki pages:

---

title: [Page Title]
type: concept | source | entity | query | gap | experiment
tags: [tag1, tag2]
sources: [count]
updated: [date]
verified_against: [source-name, YYYY-MM-DD]   # concept pages only
confidence: high | medium | low               # concept pages only

---

## [Title]

**Summary**: One sentence.

[Main content with [[wikilinks]] to related pages]

## Related

- [[Page Name]]

## On Notebooks (.ipynb)

When reading or updating wiki entries for notebook sources:
- **Never rely solely on context-mode indexing** — it misses sections whose headers use `##` instead of `**`
- Always verify the full section structure by running: `jq -r '.cells[] | select(.cell_type=="markdown") | .source[0]' notebook.ipynb` or equivalent Python to list all markdown cell first lines
- Only after confirming the real structure should you update the wiki
- **When updating from a new version:** for every detail present in the old wiki, explicitly verify it still exists in the new notebook before keeping it — do not assume anything carried over

## On Research Topic Suggestions

Whenever I ask for research topics (any phrasing: "suggest topics", "give me ideas", "what could we publish", etc.), follow this process **before proposing anything**:

1. **Do not propose topics from intuition alone.** Every proposed topic must pass a thorough web-search + prior-work scan first.
2. **Scan required per topic (minimum 4 searches):**
   - Exact keyword match ("X on small LMs", "Y placement comparison")
   - Near-synonym search (terms the authors would actually use)
   - Venue-specific search (arXiv 2024-2026, EMNLP/ACL/NeurIPS/ICLR, relevant journals)
   - Negative-result / reproduction search (to check if the finding is already known)
3. **Report findings honestly:** If a topic has prior work that covers the claimed contribution, say so and kill the topic. Do not paper over prior work with minor axis twists.
4. **Each surviving topic must have:**
   - At least 2 concrete contributions (method + empirical, or empirical + practical)
   - Feasibility under the user's stated budget (GPU VRAM, time, single person)
   - Explicit prior-work citations showing the novel delta
   - A realistic target venue (name it; don't say "some journal")
5. **Venue realism:** For UGC-CARE / low-tier Scopus / Indian journals, the novelty bar is lower — reproduction-on-small-scale is acceptable if reframed with explicit prior-work citations. State this positioning openly.
6. **UGC-CARE verification (mandatory):** Never trust a journal's own website, Scopus listing, or any third-party site claiming UGC-CARE indexing. Always verify directly against the official UGC-CARE list at https://ugccare.unipune.ac.in. A journal claiming UGC-CARE status without appearing on that list must be treated as unverified and flagged to the user.
7. **Never propose a topic if I am not confident it will survive peer review.** The user has said: "I don't want to repent later after a month of work." Treat every proposal as if the user will start work immediately.
8. **Output format when proposing:** For each topic → (a) one-line summary, (b) prior-work scan summary with citations, (c) novel delta, (d) 2 contributions, (e) feasibility numbers, (f) target venue, (g) confidence rating.

## Learning Path (learning-path.md)

`learning-path.md` defines the pedagogical reading order across all concept pages in 7 stages: Foundations → Transformer Internals → GPT-2 Architecture → Training Mechanics → Inference & Decoding → Fine-Tuning & Adaptation → Evaluation & Scaling. It has a `## Gaps in This Path` section listing topics that belong in the sequence but have no page yet.

- On every ingest: insert new concept pages in the right stage; remove filled gaps; add new gap placeholders
- On every lint run: verify all concept pages appear somewhere in the path

## On Slide Decks

Deliver slide decks as a single `.md` file. Never generate .pptx, build scripts, or image exports.

**Slide 1 is always a Cover slide** with this exact structure:

```
## Slide 1 — Cover

# [Presentation Title]

*[Subtitle]*

**Dr. Navdeep Singh**
ASSOCIATE PROFESSOR
*Computer Science & Engineering · Punjabi University, Patiala*
```

**All other slides:**
- Start with `## Slide N — Title`
- Content in plain bullet points — easy to understand, not jargon-heavy
- No tables ever
- Write so external apps (Gamma, Gemini, Claude design, etc.) can render it into a visual deck

When producing a slide deck based on wiki content, follow learning-path.md stage order. Skip any stage listed under `## Gaps in This Path` (silently).

## On Experiments (wiki/experiments/)

For active research papers, track experiment runs as structured files: `wiki/experiments/[short-name].md`

Required fields per experiment file:
- `hypothesis:` what you expect to find
- `config:` key hyperparameters / setup
- `result:` outcome (numbers, observations)
- `conclusion:` what it means; what to try next

log.md remains a timeline; experiments are queryable by outcome.

## Claim Confidence Tagging

Facts in concept pages may carry inline confidence tags:
- `[well-established]` — consensus across multiple sources
- `[contested]` — sources disagree; flag the conflict
- `[single-source]` — only one source supports this; treat with caution

During Q&A, surface these tags explicitly rather than presenting contested claims flatly.

## Last Verified Field

Concept pages carry a `verified_against:` frontmatter field — the source name and date the page was last checked against a raw source. When a raw source is deleted or updated, treat all concept pages with `verified_against: [that source]` as potentially stale and flag them.

## Dense Storage Format

Wiki pages are notes for the LLM, not explanations for humans. Store information densely:
- Use tables, bullet key:value pairs, code snippets
- No prose padding, no analogies, no "think of it like..." sentences
- The LLM reconstructs full human-friendly explanations from dense notes at query time
- When answering the user, explain in simple, easy-to-understand language
- Aim for maximum information per line; cut any line that restates another

What to preserve as-is:
- **ASCII diagrams** — keep exactly; they convey shape/flow/connections more densely than any alternative
- **Code snippets** — already dense; do not paraphrase
- **Shape traces and math** — already dense; do not paraphrase

---

## On Self-Improvement

**Auto-update `feature-menu.md`:** Whenever a new feature is added to this wiki (new workflow, new rule, new structural pattern), regenerate `feature-menu.md` immediately to reflect the current state.

**Proactive suggestions:** Whenever a pattern of weakness is noticed — repeated gaps in the same area, stale concept pages, workflow friction, structural inconsistency — surface it to the user proactively with a concrete suggestion. Do not wait to be asked. Keep suggestions short: one sentence on the problem, one sentence on the fix.
