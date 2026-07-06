# Lessons

Permanent record of process corrections and anti-patterns for this repo. Read at session start (see `CLAUDE.md`). Append here immediately when corrected on a process mistake — never let a correction live only in chat.

Keep terse — one line per rule, no elaboration; consolidate overlapping rules rather than appending near-duplicates.

## Process rules
*(none yet — append here when corrected on a process mistake)*

## Anti-patterns
- Never edit `raw/` — immutable source data.
- Never create a file under `wiki/` without linking it in `index.md` in the same session — orphans accumulate silently otherwise (see `orphan-check.sh`).
