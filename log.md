## [2026-05-09] ingest | raw/instruction_fine_tuning.py (updated version)

- Updated: wiki/sources/instruction_fine_tuning.md (EOS stop token, evaluation section)
- Updated: wiki/concepts/instruction-fine-tuning.md (EOS stop token, LLM-as-judge pipeline)
- Updated: index.md summary
- New content: generate() stops at token 50256; full Groq evaluation pipeline added
- No conflicts found

## [2026-05-09] remove | research pages moved to separate Research wiki

- Deleted: wiki/queries/research-p3-sparse-lora.md
- Deleted: wiki/queries/research-p3-study-guide.md
- Deleted: wiki/queries/research-student-hinglish-lora.md
- Deleted: wiki/queries/research-student-layer-importance-comparison.md
- Updated: index.md (removed P3, S1, S2 entries)

## [2026-05-08] update | concept: Instruction Fine-Tuning — added performance improvement strategies

- Updated: wiki/concepts/instruction-fine-tuning.md (added 5 strategies: hyperparameter tuning, more data, prompt engineering, larger model, PEFT/LoRA)
- Source: screenshot from instruction_fine_tuning notebook

## [2026-05-07] ingest | Raschka S. - Build a Large Language Model - 2025.pdf

- Created: wiki/sources/Raschka-LLM-2025.md
- Created concepts: cosine-decay.md, gradient-clipping.md, zero-shot-few-shot.md, bert.md
- Created entity: sebastian-raschka.md
- Updated concepts: weight-tying.md (Raschka caveat on separate layers), lr-warmup.md (Appendix D formula), instruction-fine-tuning.md (Shi et al. 2024 citation, contested tag), large-language-models.md (GPT-3 data table, zero/few-shot)
- Updated: index.md (6 new entries), learning-path.md (4 new concepts, 1 gap closed)
- No significant conflicts found; 1 minor update (weight tying caveat), 1 contested flag added (instruction masking)

## [2026-05-07] create | concept: LLM Evaluation + query: How do we evaluate LLMs?

## [2026-05-05] update | CLAUDE.md — 11 new wiki features adopted

- Added: Features 1–8 from feature-menu.md (concept page updates on Q&A, breadcrumb/artifact query types, gap tracking, learning path, slide deck order, contradiction checking on ingest, source transparency, structural-only lint)
- Added: Q2-A claim confidence tagging, Q2-B verified_against field, Q2-C experiment log
- Backfilled: verified_against + confidence on all 34 concept pages
- Created: learning-path.md (7 stages, 34 concepts placed, 10 gaps listed)
- New directories: wiki/gaps/, wiki/experiments/

## [2026-05-05] fix | Lint fixes — orphans + missing concept pages

- Fixed orphan: instruction-finetuning-collate-padding-trick (backlink added to data-pipeline)
- Fixed orphan: why-concept-pages (backlink added to fine-tuning.md)
- Created: wiki/concepts/instruction-fine-tuning.md
- Created: wiki/concepts/cross-entropy-loss.md
- Updated: index.md, log.md

## [2026-05-05] lint | Lint 2026-05-05

- Created: wiki/lint/lint-2026-05-05.md
- Findings: 2 orphans (collate-padding-trick, why-concept-pages), 2 raw .py files need re-ingest check, 2 missing concept pages (Instruction Fine-Tuning, Cross-Entropy Loss)
- Updated: index.md, log.md

## [2026-05-01] query | Instruction Fine-Tuning — Collate Padding Trick

- Created: wiki/queries/instruction-finetuning-collate-padding-trick.md
- Updated: index.md

## [2026-04-30] query | Instruction Fine-Tuning — Data Preparation Pipeline

- Created: wiki/queries/instruction-finetuning-data-pipeline.md
- Updated: index.md

## [2026-04-30] query | Instruction Fine-Tuning — Training Mechanics

- Created: wiki/queries/instruction-finetuning-training-mechanics.md
- Updated: index.md

## [2026-04-30] lint | lint-2026-04-30

- Fixed: instruction-finetuning-prompt-format orphan (added backlink)
- Fixed: index.md duplicated description + stale "Cell 45 & 56" reference
- Created: wiki/entities/stanford-alpaca.md
- Updated: index.md (new lint entry, new entity entry)
- Gaps flagged: instruction-finetuning concept page, cross-entropy, quantization

## [2026-04-30] query | Instruction Fine-Tuning — Prompt Format (Alpaca)

- Created: wiki/queries/instruction-finetuning-prompt-format.md
- Updated: index.md

## [2026-04-30] query | Instruction Fine-Tuning — Data Format

- Created: wiki/queries/instruction-finetuning-data-format.md
- Updated: index.md

## [2026-04-30] ingest | gpt2_decoder.py + classification_fine_tuning.py (replaced GPT2_Clean.ipynb)

- Deleted source: raw/GPT2_Clean.ipynb
- New sources: raw/gpt2_decoder.py, raw/classification_fine_tuning.py
- Renamed: wiki/sources/GPT2_Clean.md → wiki/sources/gpt2_decoder.md
- Updated all references: index.md, lr-warmup.md, gpt2-parameter-count.md, context-length-assert.md, inference-sliding-window.md
- Created: wiki/sources/classification_fine_tuning.md
- Updated: wiki/concepts/fine-tuning.md (head replacement, freeze strategy, drop_rate=0, last-token logits)
- Updated: wiki/concepts/gpt2-from-scratch.md (sources count)
- Updated: wiki/queries/inference-sliding-window.md (removed notebook cell references)
- Updated: index.md

## [2026-04-26] query | Dropout During Fine-Tuning — Why Set drop_rate=0.0

- Created: wiki/queries/dropout-during-finetuning.md
- Updated: index.md, log.md

## [2026-04-24] query | Train vs Val vs Test Split — Why All Three?

- Created: wiki/queries/train-val-test-split.md
- Updated: index.md

## [2026-04-22] query | DataLoader parameters — shuffle and drop_last

- Created: wiki/queries/dataloader-parameters.md
- Updated: index.md

## [2026-04-24] query | requires_grad vs torch.no_grad()

- Created: wiki/queries/requires-grad-vs-no-grad.md
- Updated: index.md, log.md

## [2026-04-24] query | Classification fine-tuning strategy — what to freeze and train

- Created: wiki/queries/classification-finetuning-strategy.md
- Updated: index.md, log.md

## [2026-04-24] update | context-length-assert — corrected assert code, added rule table

- Updated: wiki/queries/context-length-assert.md — corrected assert to use max_tokens + GPT_CONFIG_124M (matches actual code); added rule table for auto vs manual max_tokens scenarios; updated title/sources
- Updated: index.md

## [2026-04-24] lint | Wiki lint run (run 7)

- Created: wiki/lint/lint-2026-04-24.md
- Fixed 11 orphan pages by adding backlinks across 8 files
- 0 broken links in active pages (broken links in old lint files are harmless artifacts)
- 4 acceptable orphans (2 source pages, 1 meta page, 1 lint archive)
- 3 knowledge gaps flagged: backpropagation, BatchNorm, kv-caching→inference-sliding-window
- Updated: index.md

## [2026-04-23] query | Inference sliding window — context length handling during generation

- Created: wiki/queries/inference-sliding-window.md — training vs inference comparison; sliding window fix; bug documented in Cell 45 & 56 of GPT2_Clean.ipynb
- Updated: index.md

## [2026-04-23] query | Context length assert — why max_length must not exceed context_length

- Created: wiki/queries/context-length-assert.md — pos_emb index out of bounds explanation; why auto-truncation doesn't happen; fix via SpamDataset truncation
- Updated: index.md

## [2026-04-23] update | GPT-2 parameter count — verified 162M from code

- Updated: wiki/queries/gpt2-parameter-count.md — confirmed assign() uses copy_() not tensor aliasing; no weight tying line in __init__; out_head always separate → always 162M in this implementation
- Verified against: raw/GPT2_Clean.ipynb

## [2026-04-23] query | GPT-2 parameter count — 124M vs 162M

- Created: wiki/queries/gpt2-parameter-count.md — full component breakdown; double-count explanation; deduplicated vs raw count code
- Updated: index.md

## [2026-04-23] update | Complete input-to-output workflow — expanded with dropout, residuals, per-block steps

- Updated: wiki/queries/input-to-output-workflow.md — added explicit dropout placements table (2 locations), per-block 11-step breakdown, FFN independence note, expanded shape trace with dropout row

## [2026-04-22] update | Fine-tuning concepts from Q&A session

- Updated: wiki/concepts/fine-tuning.md — added PEFT family clarification (LoRA+QLoRA are 2 of many; full family includes Prefix Tuning, Prompt Tuning, Adapters, DoRA, AdaLoRA, Sparse LoRA); added max_tokens consistency rule with code example
- Created: wiki/queries/spam-dataset-implementation.md — full SpamDataset PyTorch pattern; tokenize→truncate→pad→tensor; dtype=torch.long requirement; max_tokens split consistency
- Updated: index.md

## [2026-04-18] replan | Student papers — killed old topics, verified new ones

- Killed (deleted) research-student-rank-sweep.md and research-student-ffn-attention-placement.md
- Reason: prior-work scan found Rathore et al. (AACL-IJCNLP Findings 2025, arXiv 2512.15634) covers rank sweep on reasoning tasks; Fomenko et al. (Microsoft 2024, arXiv 2404.05086) covers FFN vs attention placement
- Created: wiki/queries/research-student-hinglish-lora.md — S1 (Hinglish placement + rank on COMI-LINGUA tasks with XLM-R/MuRIL)
- Created: wiki/queries/research-student-layer-importance-comparison.md — S2 (IST vs Act-LoRA vs Fisher vs similarity methods on small LMs)
- Both verified against 12 WebSearches; explicit novel delta per plan's Prior Work Scan table
- Added CLAUDE.md rule: all future research suggestions must pass 4-search prior-work scan with named-citation novel delta before proposing
- Sequence: S1 first (India-niche, simpler infra), S2 after (reuses pipeline, bridges to P3 LOLO)

## [2026-04-18] plan | 2 student UGC research papers (companions to P3) [SUPERSEDED]

- Created: wiki/queries/research-student-rank-sweep.md — Student Paper 1 (rank sensitivity, 4–5 weeks); do first
- Created: wiki/queries/research-student-ffn-attention-placement.md — Student Paper 2 (FFN vs Att placement, 5–6 weeks); do after Paper 1
- Both are safe carve-outs from P3 (different axes); both serve as building blocks for main paper
- Sequence rationale: Paper 1 teaches PEFT basics with minimum custom code, produces fast first publication, infrastructure transplants to Paper 2

## [2026-04-18] create | 5 missing concept pages (from lint gaps)

- Created: wiki/concepts/lora.md — LoRA: rank decomposition, QLoRA/AdaLoRA variants, placement question → P3
- Created: wiki/concepts/temperature.md — temperature scaling in decoding pipeline
- Created: wiki/concepts/softmax.md — formula, uses in attention + output head, numerical stability
- Created: wiki/concepts/perplexity.md — exp(loss), intuition, limitations, relationship to val_loss
- Created: wiki/concepts/lr-warmup.md — linear warmup + cosine decay; note GPT-2 notebook uses no warmup

## [2026-04-18] lint | wiki lint run

- Created: wiki/lint/lint-2026-04-18.md
- 0 orphans; 7 backslash-malformed wikilinks + 6 title-case links flagged; 5 persistent gaps (LoRA, temperature, softmax, perplexity, warmup); no contradictions

## [2026-04-18] update | GPT2_Clean.ipynb — reflect new notebook (72 cells)

- Updated: wiki/sources/GPT2_Clean.md — corrected cell count (74→72); expanded section 10 inference into Temperature/Multinomial/Top-k subsections; split Fine-Tuning into sections 12+13; updated inference pipeline detail
- Updated: wiki/sources/GPT2_Clean.md Key Implementation Details — inference loop now documents full pipeline with top-k/temperature/multinomial roles
- Updated: CLAUDE.md — added rule to always verify notebook structure directly (not rely on context-mode indexing)

## [2026-04-18] update | P3 venue table — annotate SCIE indexing status

- Updated: wiki/queries/research-p3-sparse-lora.md — Target Venues section now shows Scopus + SCIE columns + approximate IFs; added Neurocomputing; added "avoid on first paper" warning for ESWA + Neurocomputing
- Updated: wiki/queries/research-p3-study-guide.md — added Scopus-vs-SCI FAQ entry
- Updated: memory/project_p3_paper.md — noted user's SCIE indexing preference (UGC/India context)

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

## [2026-05-06] ingest | Instruction Fine-Tuning (Notebook)

- Created: wiki/sources/instruction_fine_tuning.md
- Updated: wiki/concepts/instruction-fine-tuning.md — fixed masking conflict (padding-only, not instruction tokens); added training hyperparameters section
- Updated: wiki/concepts/decoding-strategies.md — added deterministic eval mode (top_k=1, temp=1)
- Updated: index.md
