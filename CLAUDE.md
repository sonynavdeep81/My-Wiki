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
- wiki/lint/ → lint run outputs (orphan checks, gap reports, contradiction flags)
- index.md → master catalog with one-line summary per page
- log.md → append-only log, format: ## [YYYY-MM-DD] action | title

## On Ingest (when I say "ingest raw/filename.md"):

1. Read the source carefully
2. Discuss key takeaways with me briefly
3. Create wiki/sources/[filename].md with summary + key points + [[backlinks]]
4. Create or UPDATE wiki/concepts/\*.md for each concept found
5. Create or UPDATE wiki/entities/\*.md for tools/people mentioned
6. Update index.md — add new pages, update summaries
7. Append to log.md

## On Query (when I ask a question):

1. Read index.md to find relevant pages
2. Read those pages fully
3. Synthesize answer with [[wikilink]] citations
4. Ask me: "Should I file this answer as a wiki page?"
5. If yes, save to wiki/queries/[topic].md

## On Lint (when I say "lint the wiki"):

- Find orphan pages (no inbound links)
- Find concepts mentioned but without their own page
- Flag contradictions between pages
- Suggest new sources to fill knowledge gaps
- Append findings to wiki/lint/lint-[date].md

## Format for wiki pages:

---

title: [Page Title]
type: concept | source | entity | query
tags: [tag1, tag2]
sources: [count]
updated: [date]

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
