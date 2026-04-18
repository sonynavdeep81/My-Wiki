## [2026-04-18] query | P3 student-friendly study guide

- Created: wiki/queries/research-p3-study-guide.md
- Plain-English companion to the technical plan; addresses intimidation factor
- Covers: problem in simple words, research gap explained, step-by-step work, 5-level study roadmap, week-by-week study pairing with experiments, honest answers to common fears
- Updated: index.md

## [2026-04-18] cleanup | Deleted superseded research files

- Removed: wiki/queries/research-topics-feasible.md (8 LLM-generated topics, novelty gaps)
- Removed: wiki/queries/research-b3-prompt-sensitivity.md (saturated prior work + 4GB mismatch)
- Rationale: decision history already captured in research-p3-sparse-lora.md §Decision Log and memory/project_p3_paper.md; retention added clutter
- Updated: index.md (removed entries), research-p3-sparse-lora.md (removed wikilinks)

## [2026-04-18] query | Research P3 — LoRA placement transferability paper plan filed

- Created: wiki/queries/research-p3-sparse-lora.md — full research plan
- Topic: cross-task layer-placement transferability for LoRA in small LMs (100M–1B)
- Locked title: "Does Layer Importance Transfer? Cross-Task Universal Placement for LoRA Fine-Tuning of Small Language Models"
- 3 contributions: LOLO protocol (method), τ transferability matrix (empirical), U^K universal placement recipe (practical)
- Target: IEEE Access (primary), Applied Sciences / NCA / ESWA (backup)
- Models (4GB fit): GPT-2 medium, Pythia-410M, TinyLlama-1.1B (4-bit)
- 8-week execution plan with ~350 GPU-hour budget and halving-fallback
- Prior-work scan (2026-04-18) covered: IST, AdaLoRA, LA-LoRA, NormAL LoRA, AlphaLoRA, Dynamic LoRA, MoDULA, SoRA, LoRA-FA, Task Arithmetic
- Mandatory baseline: IST (EMNLP Findings 2024); fail-safe: informative-negative-result framing if transferability breaks
- Superseded: research-topics-feasible.md, research-b3-prompt-sensitivity.md (marked in index.md)
- Memory updated: project_p3_paper.md + MEMORY.md index

## [2026-04-17] query | Research B3 — strengthened novelty, 4-model set, domain-specific

- Updated: wiki/queries/research-b3-prompt-sensitivity.md
- Added: prior work disclosure (Zhao 2021, Lu 2021, Webson & Pavlick 2022) + novelty gap analysis
- Added: C4 contribution — cross-model sensitivity curve across 4 sizes
- Updated model set: GPT-2 Large + Phi-3 Mini 3.8B + Mistral 7B (4-bit) + LLaMA 3 8B (4-bit)
- Added: hardware table, 4-bit quantization code (BitsAndBytesConfig), 2 domain tasks (medical, legal)
- Updated: timeline 6→7 weeks, paper structure reflects 3 results tables
- Updated: index.md

## [2026-04-17] query | Research B3 full implementation guide

- Created: wiki/queries/research-b3-prompt-sensitivity.md
- Covers: core idea, metrics explanation, 3 contributions, step-by-step code, analysis, paper structure, 6-week timeline
- Updated: index.md

## [2026-04-17] query | Expanded research topics with Scopus journal targets

- Updated: wiki/queries/research-topics-feasible.md
- Added 6 Scopus journal topics (B1–B6) with 3 explicit novel contributions each
- Kept 2 existing conference topics (A1–A2); added Part A/B structure
- Added conference vs journal distinction, contribution types, full comparison table
- Updated: index.md

## [2026-04-17] compress | Aggressive compression of verbose pages

- Rewrote to dense format: training-loop-primitives, why-save-optimizer-state, input-to-output-workflow, multi-head-attention, llm-evaluation-metrics
- Line count reductions: 117→47, 88→43, 140→57, 96→53, 101→45
- Updated CLAUDE.md: added Dense Storage Format rule
- No information lost; analogies/padding removed

## [2026-04-16] lint | Wiki Lint (run 5)

- Fixed 6 broken wikilinks: title-case slugs in pytorch-nn-building-blocks, escaped-pipe in training-loop-primitives
- 0 orphan concepts, 0 orphan entities
- Under-linked: label-smoothing (1 inbound only)
- Gaps flagged: softmax, temperature, LoRA, perplexity, warmup, backpropagation
- Created: wiki/lint/lint-2026-04-16.md
- Updated: index.md

## [2026-04-16] restructure | Move lint files to wiki/lint/

- Created: wiki/lint/
- Moved: lint-2026-04-14.md, lint-2026-04-14b.md, lint-2026-04-14c.md, lint-2026-04-14d.md
- Updated: index.md (new ## Lint section), log.md paths, CLAUDE.md directory structure + lint rule

## [2026-04-16] query | Why Save the Optimizer State?

- Created: wiki/queries/why-save-optimizer-state.md
- Updated: index.md

## [2026-04-16] query | Training Loop Primitives

- Created: wiki/queries/training-loop-primitives.md
- Updated: index.md

## [2026-04-16] query | Bias Comparison — GPT-2 vs Attention Is All You Need

- Created: wiki/queries/bias-comparison-gpt2-vs-paper.md
- Updated: index.md

## [2026-04-14] lint | Wiki Lint (run 4)

- Fixed 2 orphans: optimizer.md (backlink from gpt2-from-scratch), label-smoothing.md (backlink from dropout)
- 0 broken wikilinks, 0 under-linked concepts
- Created: wiki/lint/lint-2026-04-14d.md
- Updated: index.md

## [2026-04-14] query | Evaluation Metrics for a Decoder-Only LLM

- Created: wiki/queries/llm-evaluation-metrics.md
- Updated: index.md

## [2026-04-14] concept | Label Smoothing, AdamW, BLEU Score stubs

- Created: wiki/concepts/label-smoothing.md, wiki/concepts/optimizer.md, wiki/concepts/bleu-score.md
- Updated: index.md

## [2026-04-14] lint | Wiki Lint (run 3)

- No orphans, no broken links, no under-linked concepts
- 7 apparent broken links all in lint-2026-04-14b.md (table artifacts, not fixed)
- Gaps flagged: Label Smoothing, AdamW, BLEU Score
- Created: wiki/lint/lint-2026-04-14c.md
- Updated: index.md

## [2026-04-14] query | model.parameters() and p.numel()

- Created: wiki/queries/model-parameters-numel.md
- Updated: index.md

## [2026-04-14] query | Why Use register_buffer?

- Created: wiki/queries/register-buffer.md
- Updated: index.md

## [2026-04-14] query | Why .bool() on the Causal Mask

- Created: wiki/queries/causal-mask-bool.md
- Updated: index.md

## [2026-04-14] query | GPT-2 vs Attention Is All You Need Parameter Comparison

- Created: wiki/queries/gpt2-vs-attention-paper-params.md
- Updated: index.md

## [2026-04-14] ingest | Attention Is All You Need (Paper)

- Created: wiki/sources/Attention_2023.md
- Created concepts: cross-attention
- Updated entities: attention-is-all-you-need (major expansion)
- Updated concepts: feed-forward-network (ReLU vs GELU, labelled by source), multi-head-attention (labelled GPT-2 vs paper numbers)
- Updated: index.md

## [2026-04-14] concept | GELU + Dropout stubs

- Created: wiki/concepts/gelu.md, wiki/concepts/dropout.md
- Added backlinks: feed-forward-network → gelu + dropout
- Updated: index.md

## [2026-04-14] lint | Wiki Lint (run 2)

- Fixed: 3 broken wikilinks in pytorch-nn-building-blocks (FFN slug, gpt2-from-scratch slug)
- Fixed: 1 orphan page (pytorch-nn-building-blocks) — added backlink from gpt2-from-scratch
- Gaps noted: Dropout, AdamW, GELU, Backpropagation have no concept pages
- Created: wiki/lint/lint-2026-04-14b.md
- Updated: index.md

## [2026-04-14] query | PyTorch nn Building Blocks

- Created: wiki/concepts/pytorch-nn-building-blocks.md
- Updated: index.md

## [2026-04-14] query | Feasible Research Topics

- Created: wiki/queries/research-topics-feasible.md
- Updated: index.md

## [2026-04-14] lint-fix | Fix Lint Issues

- Created concepts: positional-embeddings, residual-connections, causal-masking
- Fixed broken wikilink: [[positional-embeddings]] now resolves
- Added backlinks: fine-tuning→llama, multi-head-attention→attention-is-all-you-need+causal-masking, llama→fine-tuning+positional-embeddings, layer-normalization→residual-connections, decoder-only-architecture→causal-masking+residual-connections
- Updated: index.md

## [2026-04-14] lint | Wiki Lint

- Created: wiki/lint/lint-2026-04-14.md
- Findings: 1 broken wikilink (positional-embeddings), 3 under-linked pages, 4 missing concept pages, 5 suggested new sources
- Updated: index.md

## [2026-04-14] query | Input Text to Output Tokens

- Created: wiki/queries/input-to-output-workflow.md
- Updated: index.md

## [2026-04-13] ingest | GPT-2 From Scratch (Notebook)

- Created: wiki/sources/GPT2_Clean.md
- Created concepts: gpt2-from-scratch, decoding-strategies, weight-tying, fine-tuning
- Created entities: gpt-family
- Updated: index.md

## [2026-04-13] ingest | Decoder Architecture (Slide Deck)

- Created: wiki/sources/Decoder_archtecture.md
- Created concepts: large-language-models, transformer-architecture, decoder-only-architecture, tokenization, byte-pair-encoding, embeddings, multi-head-attention, layer-normalization, feed-forward-network, scaling-laws, emergent-abilities, kv-caching, inference-scaling
- Created entities: attention-is-all-you-need, tiktoken, llama
- Updated: index.md
