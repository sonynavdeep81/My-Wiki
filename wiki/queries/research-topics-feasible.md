---
title: Feasible Research Topics for Conference Publication
type: query
tags: [research, probing, fine-tuning, lora, interpretability, emergence]
sources: 2
updated: 2026-04-14
---

## Feasible Research Topics for Conference Publication

**Summary**: Two publishable research directions grounded in LLM internals — both doable with 1–2 GPUs using open-source models, no pretraining required.

---

## Topic 1: Layer-wise Emergence of Linguistic Capabilities

### Research Question
The decoder-only transformer literature claims early blocks learn syntax, middle blocks learn semantics, and late blocks learn reasoning (see [[decoder-only-architecture]]). But this is qualitative. Where *exactly* do specific capabilities emerge across layers, and does the threshold shift with model size?

### Methodology
- Take pretrained models of different sizes (GPT-2 small/medium/large, LLaMA 3 8B/70B)
- Attach lightweight **probing classifiers** to each intermediate layer's hidden states
- Train probes on labeled datasets:
  - Syntax: POS tagging, dependency parsing (Penn Treebank)
  - Semantics: NLI, word sense disambiguation (SNLI, WSD benchmarks)
  - Reasoning: arithmetic, commonsense (BIG-Bench subsets)
- Plot probe accuracy vs. layer depth across model sizes

### Why It's Feasible
- No pretraining — use off-the-shelf [[llama]] and [[gpt-family]] checkpoints
- Probes are tiny (linear classifiers on top of frozen hidden states)
- All datasets are public
- 1 GPU sufficient

### Likely Venues
- BlackboxNLP @ EMNLP
- RepL4NLP workshop
- ICLR workshop on mechanistic interpretability

### Connection to Wiki
- [[emergent-abilities]] — quantifies the "where does it emerge" question
- [[decoder-only-architecture]] — the layer hierarchy claim being tested
- [[scaling-laws]] — emergence threshold shifting with model size

---

## Topic 2: Few-Shot LoRA vs Full Fine-Tuning on Classification Tasks

### Research Question
[[fine-tuning]] covers LoRA as a parameter-efficient alternative to full fine-tuning. Open question: *how few labeled examples does LoRA need to match (or beat) full fine-tuning?* Does the crossover point vary by task difficulty or domain?

### Methodology
- Pick 3–4 classification benchmarks (spam, sentiment, topic classification, textual entailment)
- Fine-tune LLaMA 3 8B with:
  - Full fine-tuning
  - LoRA at ranks r = 4, 8, 16
- At labeled data sizes: 16 / 64 / 256 / 1024 examples per class
- Measure: accuracy, convergence speed, peak GPU memory, inference latency

### Why It's Feasible
- All open-source; HuggingFace PEFT library handles LoRA in ~10 lines
- Public datasets (SST-2, AG News, SNLI, SMS Spam)
- Runs on a single A100 or 24GB consumer GPU with 4-bit quantization (QLoRA)
- Clear, reviewable evaluation setup

### Likely Venues
- EMNLP findings track
- NAACL
- Efficient NLP workshop

### Connection to Wiki
- [[fine-tuning]] — the core concepts being studied
- [[large-language-models]] — base models used
- [[llama]] — primary model for experiments
- [[scaling-laws]] — data efficiency as a function of model/rank size

---

## Recommendation

| | Topic 1 | Topic 2 |
|---|---|---|
| Novelty | Higher (interpretability is hot) | Medium (practical contribution) |
| Risk | Medium (story depends on findings) | Low (evaluation is always clean) |
| Best for | First paper aiming for workshop | First paper aiming for main track |

**If this is a first paper**: Topic 2 is lower risk — the contribution is clearly useful, evaluation is unambiguous, and reviewers in the NLP community consistently value empirical efficiency studies.

## Related

- [[emergent-abilities]]
- [[decoder-only-architecture]]
- [[fine-tuning]]
- [[scaling-laws]]
- [[llama]]
