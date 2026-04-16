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
