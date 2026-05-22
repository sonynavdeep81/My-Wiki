# Formatting Rules (load when writing/reformatting wiki pages)

## Markdown rendering (Obsidian + Markdown Preview Enhanced — both KaTeX)
- **Math = LaTeX, never backticks.** Inline `$...$`; block `$$...$$` on its own lines. KaTeX-safe: standard commands; `\begin{aligned}` not `\begin{align}`; no `\mathbb` for plain text; no custom macros.
- **Backticks = code only** — Python identifiers, kwargs, paths, config vars (`bias=True`, `emb_dim=768`, `torch.nn.GELU`). Codebase variable → backticks; math symbol/formula → LaTeX.
- **Greek/symbols** ($\mu,\sigma,\gamma,\beta,\epsilon,\sqrt{},\times,\approx,\rightarrow$, sub/superscripts) → LaTeX, never raw Unicode in backticks.
- Headings/bullets/tables: clean Markdown (one `#` per layer, hyphen bullets, pipe tables).
- ASCII diagrams, code blocks, shape traces → verbatim, never LaTeX-ify.
- Blank line around block equations/tables; no trailing whitespace; one statement per line in math blocks.
- In doubt: could appear in a paper → math; could appear in a `.py` file → code.

## Storage format by file type
- `wiki/concepts/`, `wiki/sources/`, `wiki/entities/` → **dense** (tables, key:value bullets, code) — LLM reads at query time, max info per line, no prose padding/analogies.
- `wiki/queries/` → **user-facing**: full readable article (A–Z, simple language, examples) OR breadcrumb (summary + links) per the On Query rule.
- Preserve verbatim everywhere: ASCII diagrams, code snippets, math/shape traces.
