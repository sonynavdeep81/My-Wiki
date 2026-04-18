---
title: Research P3 — Cross-Task Layer Placement Transferability for LoRA in Small LMs
type: query
tags: [research, lora, peft, layer-placement, transferability, small-language-models, scopus, method-paper]
sources: 15
updated: 2026-04-18
---

## Research P3 — Cross-Task Layer Placement Transferability for LoRA in Small Language Models

**Summary**: Independent master's-thesis-adjacent Scopus paper. Investigates whether layer-importance rankings for LoRA adapter placement transfer across NLP tasks in small language models (100M–1B params), and proposes a "universal placement" recipe derived from the transfer matrix. 2–3 month timeline on single 4GB GPU.

---

## Locked Title

**"Does Layer Importance Transfer? Cross-Task Universal Placement for LoRA Fine-Tuning of Small Language Models"**

- Question-form title — robust to positive or negative empirical outcome
- "Small Language Models" signals the 100M–1B regime (the novelty axis)

---

## Core Research Question

> Given the best K-layer subset for LoRA on task A, how well does that same placement perform on task B? Can we derive one fixed placement that works across a family of related tasks, eliminating per-task search?

**Central hypothesis:**
Layer-importance rankings learned on one task partially transfer to related tasks in small LMs, allowing a "universal placement" that matches per-task search within ≤3% accuracy at 0 search cost.

**Fallback (if hypothesis fails):** The paper publishes as an *informative negative result* — "per-task placement search is necessary in small LMs, contradicting the implicit assumption in prior PEFT literature." Still Scopus-publishable at IEEE Access / Applied Sciences tier.

---

## Prior Work (comprehensive scan, 2026-04-18)

### Directly Adjacent — Must Cite + Differentiate

| Paper | Year | Claim | How we differ |
|---|---|---|---|
| **IST — Layer-wise Importance Matters** (Yao et al., EMNLP Findings 2024) | 2024 | Dynamically selects important layers during LoRA fine-tuning; per-task ranking | IST ranks per-task; we test *whether rankings themselves transfer* — orthogonal claim. IST is our primary baseline. |
| **AdaLoRA** (Zhang et al., ICLR 2023) | 2023 | Adapts *rank* per weight matrix via SVD | Adapts rank; we adapt *placement*. Different knob. Cite as foundational. |
| **LA-LoRA** (ScienceDirect 2025) | 2025 | Layer-wise adaptive rank allocation | Same as AdaLoRA axis; we do placement + transferability |
| **NormAL LoRA** (EMNLP Findings 2025) | 2025 | L2-norm as layer-importance surrogate | Post-hoc pruning; we do pre-training placement selection + transfer |
| **AlphaLoRA** (EMNLP 2024) | 2024 | Assigns LoRA experts by layer training quality | Expert assignment in MoE; we do single-LoRA placement |
| **Dynamic LoRA** (ACM 2025) | 2025 | Dynamic layer-importance evaluation, reallocation | Per-task dynamic; we do cross-task static transfer |
| **SoRA** (Sparse Low-rank Adaptation, 2023) | 2023 | Gated proximal rank pruning | Rank pruning; not placement |
| **LoRA-FA** (2023) | 2023 | Freeze A, train B; memory savings | Different dimension; cite for PEFT context |
| **MoDULA** (EMNLP 2024) | 2024 | Mixture of domain + universal LoRA experts; uses different layer sets per model | Uses different placements per model empirically; does NOT claim transferability as primary result |

### Task-Transfer / Task-Arithmetic Background

| Paper | Year | Relevance |
|---|---|---|
| **Task Arithmetic** (Ilharco et al., ICLR 2023) | 2023 | Task vectors add/subtract across tasks — supports the idea that task structure is partially shared in weight space |
| **Task Arithmetic in Tangent Space** (NeurIPS 2023) | 2023 | Refinement of Ilharco; supports linear mode connectivity |
| **Adapters Selector** (COLING 2025) | 2025 | Cross-domain multi-task LoRA *routing*; orthogonal to placement transferability |

### Small-Model / Small-Tokenizer Regime Papers (frames novelty axis)

| Paper | Year | Relevance |
|---|---|---|
| **Small Language Models are Good Too** (LREC-COLING 2024) | 2024 | Establishes 77M–40B regime is worth studying; almost all PEFT work targets 7B+ |
| **TokSuite** (arXiv 2512.20757, 2025) | 2025 | Tokenizer impact benchmark; establishes that small-model regime has distinctive behavior |

### Gap This Paper Fills

No paper published 2021–2026 claims **"layer-importance rankings learned for LoRA on one task transfer to related tasks"** as a primary contribution. IST, LA-LoRA, Dynamic LoRA all rank per-task. MoDULA uses different placements per model without claiming transferability. The cross-task transfer matrix + universal-placement recipe + small-model regime is an uncovered intersection.

---

## Three Novel Contributions

### C1 — Leave-One-Layer-Out (LOLO) LoRA Transferability Protocol (Method)

A reproducible protocol to measure layer importance *and* cross-task transferability in a single framework.

**Procedure:**
1. For each layer ℓ ∈ {1,…,L}: train LoRA with adapter placed *only* at layer ℓ (rank r=8) on task T_i; record held-out accuracy a_{i,ℓ}
2. Rank layers for task i: π_i = argsort(a_{i,·}) (descending)
3. For every task pair (i, j): compute transferability coefficient τ_{i,j} = Spearman(π_i, π_j)
4. Define top-K placements P^K_i = top-K layers from π_i
5. Measure cross-task application: Acc(P^K_i applied to task j) vs. Acc(P^K_j applied to task j)

**Why new:** IST uses gradient-based importance during training; LOLO uses direct held-out accuracy → a cleaner, model-agnostic signal. IST does not compute τ or cross-apply placements.

### C2 — Cross-Task Transferability Matrix for Small LMs (Empirical)

First published transferability matrix for LoRA placement in the 100M–1B parameter regime.

**Deliverable:** 6×6 matrix of τ_{i,j} values across 6 NLP tasks, replicated across 3 small models (GPT-2 medium, Pythia-410M, TinyLlama-1.1B).

**Interpretation key:**
| τ range | Meaning |
|---|---|
| τ > 0.7 | Strong transfer — placement is task-agnostic for this pair |
| 0.4–0.7 | Partial transfer — shared structure |
| < 0.4 | Poor transfer — per-task search needed |

**Expected (hypothesized) finding:** τ correlates with task-family similarity (sentiment ↔ topic: high; sentiment ↔ NLI: low); classification tasks cluster higher than generation tasks.

### C3 — Universal Placement Recipe (Practical)

A fixed layer subset derived from the transferability matrix that matches per-task search quality within a measured tolerance, at zero search cost.

**Derivation:**
- Aggregate ranking π* = consensus of {π_i} via Borda count across all tasks
- Universal placement U^K = top-K layers of π*
- Empirical test: for each task j, measure Acc(U^K on j) vs Acc(P^K_j on j); report mean degradation Δ

**Deliverable:** Per-model U^K recommendation (e.g., "For GPT-2 medium, place LoRA at layers [3, 5, 8, 11] — within 2.1% of per-task-optimal placement at K=4").

**Practitioner value:** Eliminates hyperparameter search cost when adapting small LMs to new downstream tasks of similar type.

---

## Methodology — Complete Spec

### Models (all fit 4GB single GPU)

| Model | Params | Layers | Format | VRAM (LoRA r=8) | Role |
|---|---|---|---|---|---|
| GPT-2 Medium | 355M | 24 | fp16 | ~1.2 GB | Primary — most layers to study |
| Pythia-410M | 410M | 24 | fp16 | ~1.4 GB | Architecture diversity check |
| TinyLlama-1.1B | 1.1B | 22 | 4-bit + LoRA | ~2.5 GB | Small-LLaMA-family representative |
| *Optional:* Phi-1.5 | 1.3B | 24 | 4-bit + LoRA | ~2.8 GB | Microsoft family; add if time permits |

**Rejected (do not fit 4GB for training):** Mistral 7B, LLaMA 3 8B, Phi-3 Mini (too tight with gradients).

**LoRA config:** r=8, α=16, dropout=0.05, targets = `q_proj` + `v_proj` (attention only — keeps scope tight; FFN placement is follow-up work).

### Tasks (6 total — covers classification + pair + generation)

| Task | Dataset | Type | Labels | Size (train/val/test) |
|---|---|---|---|---|
| T1 Sentiment | SST-2 | Single-text classification | 2 | 67k / 872 / 1.8k |
| T2 Topic | AG News | Single-text classification | 4 | 120k / — / 7.6k |
| T3 NLI | SNLI | Pair classification | 3 | 550k / 10k / 10k |
| T4 Paraphrase | MRPC | Pair classification | 2 | 3.7k / 408 / 1.7k |
| T5 QA-style | BoolQ | Pair classification | 2 | 9.4k / 3.3k / — |
| T6 Domain sentiment | IMDb | Single-text classification | 2 | 25k / — / 25k |

**Rationale:** 6 tasks span (a) single-text vs pair-text, (b) different label counts, (c) general vs long-form (IMDb). Transferability across task *types* is the interesting axis.

**Train subset per task:** 4,000 examples (keeps per-run compute manageable on 4GB; LoRA overfits smaller sets cleanly).

### LOLO Measurement Protocol (formal)

```
for model M in {GPT-2-med, Pythia-410M, TinyLlama-1.1B}:
    for task T in {T1..T6}:
        for layer ℓ in {1..L_M}:
            for seed s in {42, 137, 2024}:
                init LoRA at layer ℓ only (r=8)
                fine-tune on 4k examples of T, 3 epochs
                eval on held-out set → a_{M,T,ℓ,s}
        π_{M,T} = argsort over ℓ of mean_s(a_{M,T,ℓ,s})
    build τ matrix: τ_{M,i,j} = Spearman(π_{M,i}, π_{M,j})
```

### Transferability Metric Details

- **Primary:** Spearman rank correlation τ_{i,j} ∈ [−1, 1]
- **Secondary:** Top-K overlap Jaccard J^K_{i,j} = |P^K_i ∩ P^K_j| / |P^K_i ∪ P^K_j|
- **Statistical significance:** paired t-test on (a_{i,ℓ}, a_{j,ℓ}) across layers; report p-values

### Baselines (MANDATORY — reviewer-proofing)

1. **Full LoRA** — adapters at all layers; upper-bound reference
2. **Random-K placement** — random K-layer subset; lower-bound null
3. **IST (Yao 2024)** — our PRIMARY baseline; run their per-task selection on our tasks; compare against our U^K
4. **Top-K by layer depth** — trivial heuristic (top-K highest / lowest layers); tests whether the pattern is just "deep layers win"
5. **AdaLoRA** — for rank-budget parity (not placement parity)

---

## Experimental Matrix (total runs)

| Phase | Dimension | Count |
|---|---|---|
| LOLO ranking | Models × Tasks × Layers × Seeds | 3 × 6 × ~24 × 3 = **~1,300 runs** |
| Cross-apply (top-K) | Models × (Task pair) × K × Seeds | 3 × 30 × 3 × 3 = **~810 runs** |
| Baselines | Models × Tasks × Baselines × Seeds | 3 × 6 × 5 × 3 = **~270 runs** |
| **Total** | | **~2,380 runs** |

**Per-run compute:** ~8–15 min on 4GB for 4k examples, 3 epochs, LoRA r=8, GPT-2 medium → ~350 GPU-hours.

**Mitigation:**
- Cut LOLO to every-other-layer first (halve count); expand only if signal unclear → drops to ~175 GPU-hours
- 350h ÷ 24h/day = ~15 days of overnight scheduling (realistic for a 2-week compute window)
- Seeds: start with 1 seed; add 2 more only on the tightly-contested comparisons

---

## Expected Results / Figure Plan

1. **Figure 1** — LOLO importance curves (accuracy vs layer ℓ) per (model, task). 3 models × 6 tasks = 18 curves, arranged as 3×6 grid. Main empirical artifact.
2. **Figure 2** — Transferability matrices τ (one per model). 6×6 heatmaps.
3. **Figure 3** — Universal-placement Pareto: mean accuracy vs K (adapter layers count), compared against per-task search and baselines.
4. **Figure 4** — Cross-model universality: does GPT-2's U^K resemble Pythia's / TinyLlama's? Placement overlap bars.
5. **Table 1** — Full accuracy × model × task × method (including baselines)
6. **Table 2** — Mean degradation Δ of U^K vs per-task search, with confidence intervals

---

## Risk Analysis (honest)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Transferability is weak (τ < 0.3 across board) | Medium | Reframe as negative result; still publishable. Pivot narrative to "why small models resist universal placement." |
| IST reviewer pushback ("you're re-running IST") | High | Explicitly re-run IST on identical tasks + show our protocol measures something different (direct accuracy vs gradient importance) |
| Compute overrun (>350h) | Medium | Halve LOLO resolution + reduce to 1 seed for LOLO phase; keep 3 seeds only for final comparisons |
| LoRA-on-single-layer too weak to separate tasks | Low | Already validated: single-layer LoRA on sentiment gives ~70% vs ~55% random on GPT-2 medium per IST's findings |
| 4GB OOM with TinyLlama + gradient | Medium | Fall back to Pythia-410M only; drop TinyLlama if needed. 2-model story still valid. |
| Paper solo-write quality | High | Budget 3 weeks minimum; ask LLM for section-by-section draft review; target 10-page Scopus format |

---

## Hardware Constraints (confirmed)

- **GPU:** Single, 4GB VRAM
- **Rules out:** Mistral 7B+, LLaMA 3 8B, any 13B+ model, any model requiring full-precision gradients on 3B+ params
- **Enables:** GPT-2 family, Pythia ≤1B, TinyLlama 1.1B (4-bit + LoRA), Phi-1.5 (4-bit + LoRA)
- **Gradient checkpointing:** enabled throughout — expect 1.3× slowdown
- **Batch size:** 4–8 for GPT-2-med, 2–4 for TinyLlama
- **Precision:** fp16 for forward; bfloat16 for LoRA params if supported

---

## 8-Week Execution Plan

### Week 1 — Setup & Pipeline
- HF Transformers + PEFT install; verify LoRA placement API (`target_modules` at specific layer indices via custom wrapper)
- Load 3 models in 4GB budget; dry-run single-layer LoRA on GPT-2 medium
- Download + preprocess 6 datasets; build `get_subset(task, n=4000)` utility
- Write eval harness with seed control
- **Deliverable:** 1 full LOLO run on (GPT-2 medium, SST-2) — sanity check that layer-importance curve has a signal

### Week 2 — LOLO Phase 1 (GPT-2 medium, 6 tasks)
- Run LOLO for all 24 layers × 6 tasks × 1 seed (every layer)
- Collect ~150 runs; analyze curves
- Build first τ matrix for GPT-2 medium; sanity-check magnitudes
- **Deliverable:** Figure 1 (partial) + first τ matrix

### Week 3 — LOLO Phase 2 (Pythia + TinyLlama)
- Repeat LOLO for Pythia-410M and TinyLlama-1.1B (every-other-layer, 1 seed)
- Expand GPT-2 seeds to 3 at the contested layers
- **Deliverable:** Full 3-model LOLO data

### Week 4 — Cross-Apply + Universal Placement
- Compute Borda-consensus π* for each model
- Run cross-apply: Acc(U^K on task j) for K ∈ {2, 4, 6, 8}
- Compare against per-task P^K_j
- **Deliverable:** Figure 3 + Table 2

### Week 5 — Baselines
- Run IST baseline (re-implement from paper or use their code)
- Run random-K, top-K-depth, AdaLoRA, full-LoRA
- Full Table 1
- **Deliverable:** Complete results tables

### Week 6 — Paper Draft (sections 1–4)
- Introduction (motivate small-LM regime + per-task search cost)
- Related Work (IST, AdaLoRA, task arithmetic)
- Method (LOLO protocol, τ metric, U^K derivation)
- Experimental Setup
- **Deliverable:** 5-page draft

### Week 7 — Paper Draft (sections 5–7) + Figures
- Results section with all tables/figures
- Discussion: when transferability works, when it fails
- Conclusion + limitations + future work (FFN placement, larger models)
- Polish figures to publication quality (matplotlib → svg)
- **Deliverable:** Full 10-page draft

### Week 8 — Revision + Submit
- Self-review + LLM-assisted section review
- Format for target venue (IEEE Access preferred — single-column, ~10 pages)
- Final proofread; compile; submit
- **Deliverable:** Submitted paper

**Buffer:** If any week slips, compress Weeks 2–3 (LOLO) by dropping Pythia. Two-model story (GPT-2 + TinyLlama) is sufficient.

---

## Target Venues (ranked by fit)

All target venues are indexed in **both Scopus and SCIE** (Web of Science Core Collection / "SCI"). Acceptance at any of them qualifies as both a Scopus paper and an SCI paper.

| Venue | Scopus | SCIE | Approx. IF | Scope fit | Acceptance notes |
|---|:---:|:---:|:---:|---|---|
| **IEEE Access** | ✓ Q1 | ✓ (since 2016) | ~3 | Strong — broad AI/NLP; empirical + method | ~30% accept; fast review (4–6 weeks) |
| **Applied Sciences** (MDPI) | ✓ Q2 | ✓ | ~2.5 | Good — applied ML venue | ~45% accept; fast review |
| **Neural Computing and Applications** (Springer) | ✓ Q1 | ✓ | ~5 | Strong — PEFT fits NN applications scope | ~25% accept; 2–4 month review |
| **Expert Systems with Applications** (Elsevier) | ✓ Q1 | ✓ | ~7+ | Moderate — prefers practitioner takeaway (U^K recipe fits) | Longer review; significantly higher bar |
| **Neurocomputing** (Elsevier) | ✓ Q1 | ✓ | ~5–6 | Moderate | Higher bar than IEEE Access |

**Primary target:** IEEE Access (SCIE + realistic acceptance for solo master's work). **Backup:** Applied Sciences (SCIE, fast, higher accept rate).

**Avoid on first paper:** Expert Systems with Applications, Neurocomputing — SCIE-indexed but the bar is high for a solo independent submission without track record. Keep them as stretch goals only.

**Why this matters for career:** SCIE-indexed publications count for UGC points (India), PhD applications, and academic hiring in most systems. A Scopus-only paper (in a non-SCIE journal) has weaker weight. All venues in the table above satisfy both criteria — so you do not need to choose between "Scopus" and "SCI" strategies.

---

## Paper Structure (10 pages, Scopus format)

| Section | Pages | Content |
|---|---|---|
| 1. Introduction | 1 | Motivate per-task search cost; preview transferability finding; 3 contribution bullets |
| 2. Related Work | 1 | IST, AdaLoRA, LA-LoRA, NormAL LoRA, Dynamic LoRA, task arithmetic; identify gap |
| 3. Method | 2 | LOLO protocol, τ metric, U^K derivation; pseudocode |
| 4. Experimental Setup | 1 | Models, tasks, hardware, hyperparameters, baselines |
| 5. Results | 3 | Fig 1–4, Table 1–2, statistical tests |
| 6. Discussion | 1 | Task-family similarity vs τ; small-model vs large-model regime speculation |
| 7. Conclusion | 0.5 | Summary + limitations + future work |
| References | 0.5 | ~25–30 citations |

---

## Success Criteria (self-check before submission)

- [ ] 3 contributions each map to a distinct table/figure
- [ ] IST run as baseline with quantitative comparison
- [ ] τ matrix has statistical significance reported
- [ ] At least one finding is non-obvious (either strong transfer where unexpected, or weak transfer refuting folk knowledge)
- [ ] U^K recipe has measurable per-model recommendation
- [ ] All experiments reproducible from released code (optional GitHub repo)
- [ ] Paper reads cleanly to someone unfamiliar with LoRA specifics
- [ ] Limitations section explicitly names: FFN not studied, rank fixed at 8, tasks are all classification, only 3 small models

---

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-04-17 | Original 8 LLM-generated topics rejected | Thin novelty; 4GB VRAM mismatch |
| 2026-04-18 | Picked P3 sparse LoRA | Method paper preference + manageable |
| 2026-04-18 | Sharpened P3 → transferability framing | IST (EMNLP 2024) covers vanilla placement |
| 2026-04-18 | Considered P2 verbalizer ensemble | Found MaVEN Oct 2024 kills it — reverted |
| 2026-04-18 | **Final: P3 transferability + small-model regime** | Narrow but genuine gap; method + empirical + practical contributions; fail-safe via negative-result framing |

---

## Related

- [[fine-tuning]]
- [[large-language-models]]
- [[gpt2-from-scratch]]
- [[multi-head-attention]]
- [[optimizer]]

---

## External References (used in prior-work scan, 2026-04-18)

- IST — [Layer-wise Importance Matters (EMNLP Findings 2024)](https://aclanthology.org/2024.findings-emnlp.109.pdf)
- AdaLoRA — [Zhang 2023](https://arxiv.org/abs/2303.10512)
- LA-LoRA — [ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S089360802500975X)
- NormAL LoRA — [EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.1074.pdf)
- AlphaLoRA — [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1141/)
- SoRA — [arXiv 2311.11696](https://arxiv.org/abs/2311.11696)
- LoRA-FA — [arXiv 2308.03303](https://arxiv.org/abs/2308.03303)
- Task Arithmetic — [Ilharco 2023](https://arxiv.org/abs/2212.04089)
- MoDULA — [arXiv 2412.07405](https://arxiv.org/html/2412.07405v1)
- Adapters Selector — [COLING 2025](https://aclanthology.org/2025.coling-main.40/)
- Small LMs for Zero-Shot — [arXiv 2404.11122](https://arxiv.org/abs/2404.11122)
