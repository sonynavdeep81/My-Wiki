---
title: Feasible Research Topics — Conference & Scopus Journal
type: query
tags: [research, probing, fine-tuning, lora, interpretability, emergence, scopus, journal, conference]
sources: 2
updated: 2026-04-17
---

## Feasible Research Topics — Conference & Scopus Journal

**Summary**: Eight publishable research directions for a master's student working independently with 1–2 GPUs. Two target NLP conference workshops; six target Scopus journals with explicit novel contributions per topic.

---

## Part A — Conference / Workshop Targets

> Novelty requirement: new method, new finding, or new benchmark. Suitable for EMNLP workshops, NAACL, ICLR workshops.

---

### A1: Layer-wise Emergence of Linguistic Capabilities

**Research Question:** Where exactly do syntax, semantics, and reasoning capabilities emerge across transformer layers — and does the threshold shift with model size?

**Methodology:**
- Attach lightweight probing classifiers to each layer's hidden states of GPT-2 (small/medium/large) and LLaMA 3 8B
- Train probes on POS tagging (Penn Treebank), NLI (SNLI), commonsense (BIG-Bench)
- Plot probe accuracy vs. layer depth across model sizes

**Feasibility:** No pretraining; probes are linear classifiers on frozen states; 1 GPU; all datasets public

**Venues:** BlackboxNLP @ EMNLP · RepL4NLP · ICLR Mechanistic Interpretability workshop

**Wiki links:** [[emergent-abilities]] · [[decoder-only-architecture]] · [[scaling-laws]]

---

### A2: Few-Shot LoRA vs Full Fine-Tuning on Classification Tasks

**Research Question:** How few labeled examples does LoRA need to match full fine-tuning — and does the crossover point vary by task difficulty?

**Methodology:**
- Fine-tune LLaMA 3 8B with full fine-tuning and LoRA (r = 4, 8, 16) on 3–4 classification benchmarks
- Vary dataset size: 16 / 64 / 256 / 1024 examples per class
- Measure: accuracy, convergence speed, GPU memory, inference latency

**Feasibility:** HuggingFace PEFT handles LoRA in ~10 lines; runs on single A100 with QLoRA; public datasets

**Venues:** EMNLP findings track · NAACL · Efficient NLP workshop

**Wiki links:** [[fine-tuning]] · [[large-language-models]] · [[llama]] · [[scaling-laws]]

---

## Part B — Scopus Journal Targets

> Novelty requirement: novel empirical finding, novel practical guideline, or correction of an existing claim. Suitable for IEEE Access, Elsevier (Expert Systems with Applications, Neurocomputing, Information Processing & Management), Springer (Neural Computing and Applications, Journal of Big Data).

**Key distinction:** Conference papers need novelty in the *method*. Journal papers need novelty in the *findings* — a systematic study that produces conclusions practitioners didn't have before.

---

### B1: Attention Head Pruning — Which Heads Actually Matter?

**Research Question:** How many of GPT-2's 144 attention heads (12 layers × 12 heads) can be pruned before performance degrades — and which heads specialize?

**Novel Contributions:**
1. A ranked importance map of all 144 heads — the first such map for GPT-2 specifically
2. An empirical pruning curve (accuracy vs. sparsity) across 3 downstream tasks, giving practitioners a concrete pruning budget
3. Evidence of head specialization (syntactic, positional, rare-word) — validating or refuting Voita et al. (2019) on a different model

**Methodology:** Mask heads one at a time, measure perplexity change; rank by importance; prune bottom k% and evaluate on classification/NLI

**Feasibility:** GPT-2-small fits on any GPU; masking is ~5 lines of PyTorch; 2–3 weeks of experiments

**Target journals:** *Neural Computing and Applications* (Springer) · *IEEE Access*

**Wiki links:** [[multi-head-attention]] · [[gpt2-from-scratch]] · [[dropout]]

---

### B2: Layer Freezing Strategies in Fine-Tuning

**Research Question:** Which layers should be frozen during fine-tuning to get the best accuracy/compute tradeoff — and does the optimal strategy depend on dataset size?

**Novel Contributions:**
1. A Pareto analysis of accuracy vs. compute across 5 freezing strategies and 3 tasks — the first controlled comparison for GPT-2
2. A practical decision rule: "freeze bottom N layers when dataset size < K examples" — directly actionable
3. Evidence on whether top layers generalize across tasks — an open question in fine-tuning literature

**Methodology:** Fine-tune on sentiment/topic/NLI with freeze strategies (none / bottom 3 / 6 / 9 / all-except-head); measure accuracy + training time + GPU memory

**Feasibility:** HuggingFace Trainer; ~50 lines of code; single GPU; 3–4 weeks

**Target journals:** *Expert Systems with Applications* (Elsevier) · *Applied Sciences* (MDPI)

**Wiki links:** [[fine-tuning]] · [[layer-normalization]] · [[gpt2-from-scratch]]

---

### B3: Prompt Sensitivity in Zero-Shot Classification

**Research Question:** How much does rephrasing a prompt affect zero-shot accuracy — and what syntactic patterns characterise winning prompts?

**Novel Contributions:**
1. A quantified sensitivity score per task — establishes the magnitude of the prompt variance problem
2. A taxonomy of prompt patterns (question / instruction / fill-in-the-blank) with average performance profiles per task type
3. Practical prompt design guidelines derived from winning patterns — directly usable by NLP practitioners

**Methodology:** Write 10 semantically equivalent prompt variants per task; run zero-shot classification on GPT-2 and LLaMA 3 8B across 4 tasks; measure variance; cluster prompts by performance; identify syntactic patterns

**Feasibility:** No fine-tuning — inference only; fastest iteration of all 6 topics; 2–3 weeks

**Target journals:** *Information Processing & Management* (Elsevier) · *IEEE Access*

**Wiki links:** [[large-language-models]] · [[decoding-strategies]] · [[gpt-family]]

---

### B4: Perplexity as an Out-of-Domain Detector

**Research Question:** Can language model perplexity reliably flag inputs outside the model's training distribution — and how does it compare to existing OOD baselines?

**Novel Contributions:**
1. First systematic evaluation of raw LM perplexity as an OOD detection signal across 4+ domain pairs (news/medical, news/legal, news/code, etc.)
2. A calibrated threshold selection method — how to set the perplexity cutoff for a target false-positive rate
3. A comparison against 2–3 existing baselines (embedding distance, vocabulary overlap) showing where perplexity wins and loses

**Methodology:** Compute GPT-2 perplexity on in-domain vs. out-of-domain corpora; set thresholds; measure precision/recall; compare baselines

**Feasibility:** Pure inference — no training at all; perplexity already computed in any training loop; 2 weeks

**Target journals:** *Journal of Big Data* (Springer) · *Neurocomputing* (Elsevier)

**Wiki links:** [[llm-evaluation-metrics]] · [[gpt2-from-scratch]] · [[large-language-models]]

---

### B5: Dropout Rate Sensitivity During Fine-Tuning

**Research Question:** Is the default dropout p=0.1 from the original transformer paper optimal for fine-tuning on small datasets — and does the optimal rate vary by dataset size?

**Novel Contributions:**
1. A dataset-size × dropout-rate interaction map — the joint effect has not been systematically measured
2. A recommended dropout schedule by dataset size: e.g. "p=0.05 for <500 examples, p=0.1 for >2000" — derived empirically
3. Evidence that the default p=0.1 is suboptimal for small-dataset fine-tuning — a corrective empirical finding

**Methodology:** Fine-tune GPT-2 on 4 datasets at sizes 100/500/2000; sweep dropout rates 0 / 0.05 / 0.1 / 0.2 / 0.3; report accuracy + overfitting curves per combination

**Feasibility:** Pure hyperparameter grid search; mechanical to run; 3–4 weeks

**Target journals:** *Computers & Electrical Engineering* (Elsevier) · *Applied Sciences* (MDPI)

**Wiki links:** [[dropout]] · [[fine-tuning]] · [[optimizer]]

---

### B6: BLEU Score vs Human Judgment in Domain-Specific Text Generation

**Research Question:** How well does BLEU correlate with human quality judgments for domain-specific text (medical, legal, technical) — and which automatic metric best substitutes human evaluation per domain?

**Novel Contributions:**
1. Domain-specific correlation coefficients between BLEU/ROUGE and human ratings — showing the metric reliability gap varies significantly by domain
2. Identification of failure modes — specific text types where BLEU systematically disagrees with human judgment
3. A per-domain metric recommendation — practical guidance for NLP evaluation pipelines when human evaluation is unavailable

**Methodology:** Fine-tune GPT-2 on 2–3 domain corpora; generate text; collect human ratings (5-point scale via survey); compute Pearson/Spearman correlation between BLEU/ROUGE and human scores across domains

**Feasibility:** Text generation is fast; human eval via small survey (classmates/MTurk) is doable; 4–5 weeks

**Target journals:** *Expert Systems with Applications* · *Language Resources and Evaluation* (Springer)

**Wiki links:** [[bleu-score]] · [[llm-evaluation-metrics]] · [[decoding-strategies]]

---

## Full Comparison Table

| ID | Topic | Venue Type | Compute | Risk | Time |
|---|---|---|:---:|:---:|:---:|
| A1 | Layer-wise emergence | Conference workshop | Low | Medium | 2–3 mo |
| A2 | LoRA vs full fine-tuning | Conference main/workshop | Low | Low | 2–3 mo |
| B1 | Attention head pruning | Scopus journal | Low | Low | 2 mo |
| B2 | Layer freezing strategies | Scopus journal | Low | Low | 2 mo |
| B3 | Prompt sensitivity | Scopus journal | Very low | Very low | 1.5 mo |
| B4 | Perplexity as OOD detector | Scopus journal | Very low | Low | 1.5 mo |
| B5 | Dropout rate sensitivity | Scopus journal | Low | Very low | 2 mo |
| B6 | BLEU vs human judgment | Scopus journal | Very low | Medium | 2 mo |

**For a first Scopus paper (2–3 months):** B3 or B4 — no training needed, clean evaluation story, fast iteration.

**For a stronger Scopus paper:** B1 or B2 — more experiments but the story is richer and more cited.

---

## Related

- [[emergent-abilities]]
- [[decoder-only-architecture]]
- [[fine-tuning]]
- [[scaling-laws]]
- [[llama]]
- [[multi-head-attention]]
- [[dropout]]
- [[bleu-score]]
- [[llm-evaluation-metrics]]
