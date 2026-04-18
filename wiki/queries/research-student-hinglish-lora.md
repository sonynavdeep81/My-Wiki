---
title: Student Paper S1 — LoRA Placement & Rank Sensitivity on Hinglish Code-Mixed Tasks
type: query
tags: [research, lora, hinglish, code-mixed, peft, multilingual, student-paper, ugc, indic]
updated: 2026-04-18
---

## Student Paper S1 — LoRA Placement & Rank Sensitivity for Hinglish Classification

**Summary**: First UGC paper for master's student. First systematic LoRA placement + rank study on Hinglish code-mixed classification tasks. Uses COMI-LINGUA benchmark (EMNLP Findings 2025) + XLM-R-base / MuRIL-base under 4GB budget. 4–5 weeks, companion to [[research-p3-sparse-lora|P3]].

---

## Prior Work Scan (2026-04-18) — What Is Already Published

| Paper | Venue | Covers | Gap left open |
|-------|-------|--------|---------------|
| Hinglish hate speech + LoRA (Springer Discover Sustainability 2025) | SCIE-adjacent | Hate speech only, DeepSeek R1/Llama3/Gemma3, single task | No placement study, no rank sweep, single task |
| COMI-LINGUA (EMNLP Findings 2025) | ACL | Dataset + baselines for LID/MLI/NER/POS | No LoRA, no PEFT study |
| Fomenko "A Note on LoRA" (Microsoft 2024, arXiv 2404.05086) | arXiv | Attention vs FFN on 7B+ English | Not Hinglish, not <300M models |
| South Asia low-resource survey (EMNLP Findings 2025) | ACL | Surveys 2024-2025 | Explicitly states: "few studies methodically examine PEFT on code-mixed" |
| OpenHathi / Hindi QLoRA | Blog/report | Hindi generation, 7B LLaMA2 | Not code-mixed classification, not small LMs |

**Novel delta:** First systematic (placement × rank) study on code-mixed Hinglish classification with <300M multilingual LMs under 4GB constraint.

---

## Prerequisites

- Python + Hugging Face Transformers
- Familiarity with tokenization (multilingual tokenizers differ from GPT-2 BPE)
- Git + LaTeX for paper writing

---

## Working Title

**"LoRA on Hinglish: A Systematic Placement and Rank Study for Code-Mixed Classification with Small Multilingual Language Models"**

---

## Core Research Question

> For Hinglish code-mixed classification tasks, (a) where should LoRA be placed (attention-only, FFN-only, or all-linear), and (b) what rank suffices, when using small multilingual LMs under 4GB GPU?

---

## Two Contributions

### C1 — Placement Study for Hinglish (Empirical)

Three-way placement comparison (attention-only / FFN-only / all-linear) across 3 code-mixed tasks on 2 models at matched rank r=8.

**Deliverable:** Table 1 — accuracy per (model, task, placement) with stderr.

### C2 — Per-Task Rank Recommendation for Hinglish (Practical)

At the best placement from C1, rank sweep r ∈ {4, 8, 16, 32} per task → recommend minimum-sufficient rank per (model, task) tuple.

**Deliverable:** Table 2 + Figure 1 — rank-accuracy curves; "practitioner recommendation box" for Hinglish under 4GB.

---

## Methodology

### Models (both fit 4GB fp16 + LoRA)

| Model | Params | Tokenizer | Reason |
|-------|--------|-----------|--------|
| XLM-RoBERTa base | 270M | XLM-R SPM | De facto multilingual baseline; 100-lang support |
| MuRIL base | 236M | MuRIL SPM | Google, Indic-specialized (17 Indian langs + transliteration) |

Both load in <1GB fp16; fine-tuning stays under 3GB with batch 16.

### Tasks (3 from COMI-LINGUA subset)

| Task | Label type | Why chosen |
|------|-----------|-----------|
| Language ID (LID) | Token-level (Hi/En/Other) | Foundational; high-data |
| Matrix-Language ID (MLI) | Sentence-level | Sentence-level classification |
| POS tagging | Token-level | Test LoRA on structured prediction |

(Dropped NER from full 5-task set to keep scope single-person feasible. Paper acknowledges in Limitations.)

### LoRA Placements Compared

| Placement | Target modules (HF names) | Params for r=8 (XLM-R) |
|-----------|---------------------------|------------------------|
| Attention-only | query, value | ~0.6M |
| FFN-only | intermediate.dense, output.dense | ~1.2M |
| All-linear | q, v, i, o | ~1.8M |

Report trainable parameter count per placement for fair reading.

### Experimental Matrix

| Dim | Values | Count |
|-----|--------|-------|
| Models | XLM-R base, MuRIL base | 2 |
| Tasks | LID, MLI, POS | 3 |
| Placements (C1 phase) | Att, FFN, All | 3 |
| Ranks (C2 phase, at best placement) | 4, 8, 16, 32 | 4 |
| Seeds | {42, 137, 212} | 3 |

- C1 runs: 2 × 3 × 3 × 3 = **54**
- C2 runs: 2 × 3 × 4 × 3 = **72** (minus overlap at r=8 best-placement = already counted → net ~54)
- **Total: ~108 runs × ~15 min = ~27 GPU-hours** (budget 40 hrs with debugging)

### Baselines

1. Frozen base + linear probe (lower bound)
2. Full fine-tuning on smallest model/task (if fits memory; if not, cite COMI-LINGUA's reported numbers)

### Training Config

- fp16 mixed precision
- Batch 16, grad accum 2 (effective 32)
- 3 epochs, AdamW, lr 3e-4, cosine decay, 100-step warmup
- LoRA: α=2r, dropout=0.05
- 4000 examples per task (if available; subsample COMI-LINGUA train split)

---

## 5-Week Plan

### Week 1 — Setup + Tokenization
- Load XLM-R + MuRIL with PEFT; verify both fit in 4GB with LoRA rank 8
- Parse COMI-LINGUA dataset; align tokenization with XLM-R / MuRIL tokenizers
- Run one sanity LID fine-tune end-to-end on XLM-R base
- **Deliverable:** working `run_experiment(model, task, placement, rank, seed)` script

### Week 2 — Placement Study (C1)
- Run 54 placement × task × seed configurations on both models
- Build preliminary Table 1
- Identify best placement per (model, task) — expected: all-linear wins, but verify

### Week 3 — Rank Sweep (C2) + Baselines
- At best placement, sweep r ∈ {4,8,16,32}
- Run linear-probe baseline on all tasks
- Build Figure 1 (rank-accuracy curves)

### Week 4 — Analysis + Draft
- Compute mean ± stderr per config; paired t-test for placement differences
- Draft Introduction, Related Work (cite COMI-LINGUA, Fomenko, Hinglish hate speech paper, IST, South Asia survey), Method, Setup
- First full paper draft (sections 1–4)

### Week 5 — Results, Discussion, Submit
- Draft Results + Discussion + Limitations + Conclusion
- Polish figures (matplotlib → svg, color-blind palette)
- Self-review + advisor review
- Format + submit to target venue

---

## Target Venues (ordered by realism)

| Venue | Type | Scopus | SCIE | Realism |
|-------|------|--------|------|---------|
| IJRASET | UGC-CARE | No | No | Very high (same-month turnaround) |
| IEEE ICCCNT (conference) | IEEE indexed | Yes | No | High (July-Aug submission) |
| Springer SN Computer Science | Scopus | Yes | No | Stretch — possible if writing clean |
| Applied Sciences (MDPI) | Indic scope unclear | Yes | Yes (Q2) | Long shot — only if results are strong |

**Start with:** IJRASET (safe) + ICCCNT 2026 (free parallel submission if conference allows).

---

## Why This Is Safe Relative to P3

- **Different language domain** (Hinglish) — P3 focuses on English classification
- **Different models** (XLM-R/MuRIL) — P3 uses GPT-2 medium / Pythia / TinyLlama
- **No transferability claim** — P3's unique contribution untouched
- **No universal placement recipe** — P3's contribution untouched

---

## Why It Is a Building Block

- Placement-comparison infrastructure (target_modules switching) → reusable in P3's LOLO
- Student learns PEFT library + multilingual tokenization handling — broadly useful
- First-paper confidence build → Student Paper S2 becomes faster

---

## Success Criteria (student checklist)

- [ ] All 3 placements × 3 tasks × 2 models reported with stderr
- [ ] Rank sweep at best placement per task
- [ ] Trainable-parameter counts reported alongside accuracy
- [ ] Linear-probe baseline included
- [ ] Paired t-test for placement differences
- [ ] Limitations: 2 models only, 3 tasks (not full COMI-LINGUA 5), Hinglish only (not other code-mixed pairs), classification only (not generation)
- [ ] Related Work cites: COMI-LINGUA, Fomenko 2024, Hinglish hate speech 2025, IST 2024, South Asia survey 2025

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| COMI-LINGUA licensing blocks download | Fallback: GLUECoS (Hindi-English subset) |
| MuRIL doesn't fit with r=32 | Drop to r=16 max on MuRIL; footnote |
| Results are boring (all placements tied) | Reframe as "LoRA is placement-robust on Hinglish" — still publishable |
| Reviewers say "dataset paper already covers baselines" | Counter: we add systematic ablation not in COMI-LINGUA |

---

## Related

- [[research-p3-sparse-lora|P3 Paper (advisor's project)]]
- [[research-student-layer-importance-comparison|Student Paper S2 (next)]]
- [[lora|LoRA concept]]
- [[fine-tuning]]
