---
title: Student Paper S2 — Layer-Importance Scoring Comparison for LoRA on Small LMs
type: query
tags: [research, lora, peft, layer-importance, empirical, student-paper, ugc]
updated: 2026-04-18
---

## Student Paper S2 — Empirical Comparison of Layer-Importance Methods for LoRA on Small LMs

**Summary**: Second UGC paper. First head-to-head comparison of 4 layer-importance scoring methods (IST/gradient, Act-LoRA/activation, Fisher, similarity-based) on small LMs (<1B) using a unified protocol. Directly supports [[research-p3-sparse-lora|P3]]'s LOLO layer-selection choice. 5 weeks, ~50 GPU hours.

---

## Prior Work Scan (2026-04-18) — What Is Already Published

| Method | Paper | Validated on | Gap |
|--------|-------|-------------|-----|
| IST (gradient-norm layer scoring) | Yao et al., EMNLP Findings 2024 | LLaMA / GPT-J / BLOOMZ (7B+) | Not on <1B |
| Act-LoRA (activation magnitude) | MDPI Information 2025 | LLaMA-7B, Mistral-7B | Not on <1B; different models than IST |
| Fisher-based importance | Various PEFT papers | Scattered model scales | Never compared head-to-head with IST / Act-LoRA |
| Similarity-based (input-output dissimilarity) | arXiv 2602.05988 | Generic transformer | Not benchmarked against above 3 |
| Unified benchmark | — | — | **Does not exist** |

**Novel delta:** First side-by-side benchmark of all 4 methods on small LMs (<1B) with shared protocol (same models, same tasks, same budget).

---

## Prerequisites

- Student has completed [[research-student-hinglish-lora|Student Paper S1]] (or equivalent LoRA pipeline experience)
- Comfort with PyTorch hooks (for activation + gradient extraction)
- Eval pipeline from S1 is reusable

---

## Working Title

**"Which Layers Matter for LoRA? A Unified Empirical Comparison of Layer-Importance Scoring Methods on Small Language Models"**

---

## Core Research Question

> When using LoRA with a limited layer budget (top-k layers only), which layer-importance scoring method (gradient-norm, activation magnitude, Fisher, or representation-similarity) best predicts final fine-tuning accuracy on small language models?

---

## Two Contributions

### C1 — Unified Benchmarking Protocol (Method)

A reproducible protocol that:
- Computes all 4 layer-importance scores on identical (model, task, data) tuples
- Selects top-k layers for each method
- Trains LoRA restricted to those layers
- Compares final accuracy, agreement between methods, and compute overhead

**Deliverable:** `layer_importance.py` utility + pseudocode in paper.

### C2 — Head-to-Head Empirical Comparison (Empirical)

Reports per-method (accuracy, scoring-cost) pair for each (model, task), plus:
- Pearson/Spearman correlation between score rankings
- Overlap ratio of top-k selected layers
- "Does best-scoring method transfer across tasks?" analysis

**Deliverable:** Table 1 (accuracy per method × model × task), Figure 1 (agreement matrix between methods), Figure 2 (score-cost Pareto).

---

## Methodology

### Models (both fit 4GB)

| Model | Params | Layers | Why |
|-------|--------|--------|-----|
| GPT-2 Medium | 355M | 24 | P3-compatible; base (no instruction tuning) |
| Pythia-410M | 410M | 24 | Architecture diversity; training data known |

### Tasks (2 — keep scope tight)

| Task | Dataset | Type |
|------|---------|------|
| Sentiment | SST-2 | Single-text classification |
| NLI | SNLI | Pair-text classification |

Training subset: 4000 examples per task (matches P3 convention).

### Layer-Importance Methods Compared

| Method | Score | Cost |
|--------|-------|------|
| Gradient-norm (IST-style) | ‖∇_layer L‖₂ averaged over calibration batch | 1 backward pass |
| Activation magnitude (Act-LoRA) | ‖activations_layer‖₂ averaged over calibration | 1 forward pass |
| Fisher diagonal | E[(∂L/∂θ_layer)²] | Multiple backward passes |
| Representation similarity | 1 - cos(input, output) per layer | 1 forward pass |

All computed on a **256-example calibration subset** of each task's training set.

### Experimental Design

For each (model, task) pair:
1. Compute all 4 importance scores on calibration batch
2. For each method → select top-k = 50% layers (i.e., top-12 of 24)
3. Train LoRA r=8, α=16, dropout=0.05 on selected layers only
4. 3 epochs, 3 seeds {42, 137, 212}
5. Baselines: LoRA-all-layers, random-k-layers, bottom-k-layers

### Experimental Matrix

| Dim | Values | Count |
|-----|--------|-------|
| Models | GPT-2 med, Pythia-410M | 2 |
| Tasks | SST-2, SNLI | 2 |
| Methods | Grad, Act, Fisher, Sim + 3 baselines | 7 |
| Seeds | {42, 137, 212} | 3 |

**Total: 2 × 2 × 7 × 3 = 84 runs × ~15 min = ~21 GPU-hours** (budget 50 hrs with debugging + scoring-compute experiments).

---

## 5-Week Plan

### Week 1 — Infrastructure
- Reuse S1 pipeline; add `compute_layer_scores(model, data, method)` utility
- Implement all 4 scoring functions — unit test each against reference (e.g., gradient norm of frozen layer = 0)
- Deliverable: `layer_importance.py` with 4 scoring backends + tests

### Week 2 — Layer-Scoring Runs (Fast Phase)
- Compute all 4 scores on both models × both tasks × 3 seeds
- Record scoring compute cost (wall time + GPU-seconds)
- Build Figure 1 (agreement matrix: are methods ranking layers similarly?)

### Week 3 — Fine-Tuning with Selected Layers
- Run 84 LoRA fine-tunes with layer-restricted adapters
- Build Table 1 (accuracy per method × model × task ± stderr)

### Week 4 — Analysis + Draft
- Compute correlation between scoring methods
- Cross-task transferability: does method A's layer-ranking on SST-2 predict its ranking on SNLI?
- Paired significance test (method-vs-method per task)
- Draft Introduction, Related Work (cite IST, Act-LoRA, Fisher PEFT, similarity paper), Method

### Week 5 — Finalize + Submit
- Draft Results, Discussion, Limitations, Conclusion
- Polish figures
- Self-review + advisor review
- Submit

---

## Target Venues (ordered by realism)

| Venue | Type | Scopus | SCIE | Realism |
|-------|------|--------|------|---------|
| Applied Sciences (MDPI) | MDPI | Yes | Yes (Q2) | Moderate — rigor required |
| IJRASET | UGC-CARE | No | No | Very high |
| IEEE ICCCNT | IEEE | Yes | No | High |
| Scientific Reports (Nature) | Nature | Yes | Yes (Q1) | Stretch — only if writing is clean |

**Start with:** IJRASET (safe) + Applied Sciences (stretch parallel).

---

## Why This Is Safe Relative to P3

- **Studies layer-importance scoring method, not layer-placement transferability** — P3's contribution untouched
- **Same axis as IST (layer selection) but different question** — P3 uses IST as fixed baseline; S2 evaluates competing methods
- **Output:** "method X predicts best" → P3 can cite this for its LOLO design choice

---

## Why It Is a Building Block for P3

- Gives P3 empirical ground for which layer-importance metric to use
- The `compute_layer_scores()` utility is reusable by P3
- If Fisher or Act-based wins, P3 can adopt it → stronger main paper
- Provides seed-variance baseline for layer-importance (P3 can use this to justify its seed count)

---

## Success Criteria (student checklist)

- [ ] All 4 methods implemented and unit-tested
- [ ] Scoring compute cost reported per method
- [ ] Agreement matrix between methods included
- [ ] 3 baselines reported (all-layers, random-k, bottom-k)
- [ ] Cross-task transferability analysis included
- [ ] Limitations: 2 models, 2 tasks, fixed k=50%, classification-only, rank-8-only
- [ ] Related Work cites: IST (Yao 2024), Act-LoRA (MDPI 2025), similarity paper, Fisher-based PEFT

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| All methods give similar accuracy (no clear winner) | Reframe as "methods are interchangeable at small scale — use cheapest" |
| One method crashes on MuRIL (not applicable since GPT-2/Pythia) | Use per-model isolation |
| Fisher is too expensive at full layer count | Diagonal approximation + calibration batch size 256 |
| Reviewers want more models | Acknowledge in Limitations; cite compute budget |

---

## Related

- [[research-p3-sparse-lora|P3 Paper (advisor's project)]]
- [[research-student-hinglish-lora|Student Paper S1 (prerequisite)]]
- [[lora|LoRA concept]]
- [[fine-tuning]]
