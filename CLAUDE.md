# LLM Wiki — Schema & Rules (core)

Loaded every turn — kept lean. Verbose templates live in `wiki/schema/` (loaded on demand). Generic session / token-hygiene / model-handoff rules: `~/.claude/CLAUDE.md` (global). Every rule here is mandatory.

## Session start
Read `STATE.md` first (focus, last decision, next action); keep it current per the global rule.

## Wiki hygiene
- Also read `LESSONS.md` at session start.
- Every new file under `wiki/` must be linked in `index.md` in the same session it's created.
- Run `./orphan-check.sh` and `./broken-link-check.sh` at session start and before every commit; fix anything found (some noise is expected on title-case display-name links — use judgment).
- When corrected on a process mistake, append the rule to `LESSONS.md` immediately.

## Dilemma rule
When uncertain about scope, intent, or which file to touch — **ask rather than assume.** A one-line question is cheaper than a misdirected edit pass. If the answer is a durable insight, add it here.

## Minimal file scope rule
**Only touch files directly relevant to what the user asked.** No opportunistic cross-reference edits. Exceptions: broken links blocking the current task; `STATE.md` and `log.md` (always updated by convention).

## Purpose
Personal research wiki on "LLM internals and NLP" — based on Raschka, *Build a Large Language Model (From Scratch)* (2025) + the user's own code notebooks. The LLM writes/maintains all `wiki/` files.

## Directory structure
- `raw/` → immutable sources, never modify
- `wiki/concepts/` (one per idea), `wiki/sources/` (one per source), `wiki/entities/` (people/tools/papers/models) → dense LLM notes
- `wiki/queries/` → user-facing answers (breadcrumb or format-artifact — see On Query)
- `wiki/lint/` → structural checks
- `index.md` (catalog), `log.md` (append-only `## [YYYY-MM-DD] action | title`), `learning-path.md` (7-stage reading order + `## Gaps in This Path`), `STATE.md` (session pointer)

## On Ingest ("ingest raw/file.md")
1. Read source. 2. **Before writing: compare new claims vs existing concept pages; report conflicts with severity (minor/significant) + a reasoned position. Never silently resolve.** 3. Discuss takeaways briefly. 4. `wiki/sources/[file].md` (summary + key points + backlinks). 5. Create/update `wiki/concepts/*` — set `confidence:` + `verified_against:`; add inline `[well-established]`/`[contested]`/`[single-source]` tags. 6. Create/update `wiki/entities/*`. 7. Gap tracking: add bullets under the right stage in `learning-path.md § Gaps in This Path`; promote/remove filled gaps. 8. Update learning-path stage placements. 9. Update `index.md`. 10. Append `log.md`.

## On Query
1. `index.md` → relevant pages → read fully. 2. Synthesize with `[[wikilink]]` citations. 3. **Source transparency — label at the claim level:** `[book]` Raschka 2025 / `[notebook]` user code (`gpt2_decoder.py`, `classification_fine_tuning.py`, `instruction_fine_tuning.py`) / `[wiki]` / `[general knowledge]` / `[web]`. Never batch-label whole paragraphs. 4. **If the answer adds depth beyond the concept page, update that page immediately** (don't wait for ingest). 5. Ask: "Should I file this as a wiki page?" If yes, pick type: **breadcrumb** (summary + links; depth lives in concept pages — default) or **format artifact** (full content; format is the value). Then insert into `## Query Reading Order` in `learning-path.md` in dependency order. Surface `[contested]`/`[single-source]` tags explicitly.

## On Lint ("lint the wiki")
Structural health only (gaps handled by gap tracking). Orphans; broken wikilinks; resolved-contradiction review; open gaps in `learning-path.md § Gaps`; verify every concept page is placed in the path. Append to `wiki/lint/lint-[date].md`.

## Page frontmatter
`title / type (concept|source|entity|query) / tags / sources / updated`; concept pages add `verified_against: [source, date]` + `confidence: high|medium|low`. Then `## Title`, `**Summary**: one sentence.`, body with `[[wikilinks]]`, `## Related`.

## On Notebooks (.ipynb)
Never trust context-mode indexing alone (misses `##`-header cells). Verify with `jq -r '.cells[]|select(.cell_type=="markdown")|.source[0]' nb.ipynb`. Updating from a new version: verify every old wiki detail still exists before keeping it.

## On Research Topic Suggestions
Before proposing anything: (1) never propose from intuition — prior-work scan first; (2) min 4 searches/topic (exact keyword / near-synonym / venue-specific / negative-result); (3) report honestly, kill scooped topics; (4) each survivor: ≥2 contributions, feasible under the user's budget (GPU VRAM, time, single person), explicit prior-work delta citations, a named venue; (5) UGC-CARE / low-tier: reproduction-on-small-scale OK if reframed with citations; (6) **UGC-CARE verification mandatory** — verify against https://ugccare.unipune.ac.in, never trust the journal's own claim; (7) never propose unless confident it survives review ("don't want to repent after a month"); (8) output per topic: one-liner / prior-work scan + citations / novel delta / 2 contributions / feasibility numbers / target venue / confidence.

## Learning Path
`learning-path.md` = 7-stage reading order (Foundations → Transformer Internals → GPT-2 Architecture → Training Mechanics → Inference & Decoding → Fine-Tuning & Adaptation → Evaluation & Scaling) + `## Gaps in This Path`. Update placements on ingest; verify all pages placed on lint.

## Confidence tagging & staleness
Inline `[well-established]`/`[contested]`/`[single-source]` on claims; surface during Q&A, never present uncertain claims as settled. `verified_against:` = source + date last checked; when a source changes/deletes, flag pages verified against it as stale.

## Formatting, storage & slides
Math = LaTeX (KaTeX-safe), backticks for code only; dense notes for concepts/sources/entities, readable articles for queries. Full rules: `wiki/schema/formatting.md`. Slide decks: single `.md`, cover + bullet slides, no tables — template `wiki/schema/slide-format.md`; follow learning-path order, skip `## Gaps` stages.

## On Self-Improvement
Keep `feature-menu.md` current when a new feature/rule/pattern is added. Surface improvement ideas proactively (weaknesses + positive), one sentence each, occasionally.
