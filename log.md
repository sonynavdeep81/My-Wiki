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
