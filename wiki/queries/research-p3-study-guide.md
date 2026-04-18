---
title: P3 Paper — Student-Friendly Study Guide
type: query
tags: [research, study-guide, lora, beginner, learning-roadmap, p3]
sources: 1
updated: 2026-04-18
companion_to: research-p3-sparse-lora.md
---

## P3 Paper — Student-Friendly Study Guide

**Summary**: Plain-English companion to [[research-p3-sparse-lora]]. Explains the problem, the gap, what to do, and what to study before starting — in simple words, without jargon. Read this first. Open the technical file only when you need specifics.

---

## Read this first

You're writing a journal paper alone, on a 4GB laptop GPU, in ~2 months. That feels scary. It shouldn't. The work is built from small, repeatable steps that any master's student who knows basic PyTorch can do. The hard part is starting. Once you're in the rhythm of training → measuring → tabulating, each week gets easier. This guide walks you through everything at beginner speed.

---

## What the paper is about — in one paragraph

When people fine-tune a language model to a new task, they usually attach little "adapter" modules (called LoRA) to *every* layer of the model. But not every layer is equally useful — some layers help a lot, others barely matter. Your paper asks one simple question: **if you figure out the best layers for one task (like sentiment classification), do those same layers work well for a different task (like topic classification)?** If yes, you can skip the expensive search every time and just use a "recipe" placement. If no, that's also an interesting finding.

---

## The problem, explained like to a friend

Imagine LoRA adapters as sticky notes. Each note you attach to a layer of the model "teaches" that layer something about your task. Today, everyone sticks notes on every layer by default. But what if only 4 out of 12 layers really need the note? You'd save a ton of memory and speed things up.

Some recent papers (IST from EMNLP 2024) already showed "not all layers matter equally" — but they re-run the search *for every task*. Your question is different: is the ranking of important layers roughly the same across tasks? If so, find it once, use it forever.

---

## What other researchers already did (plain-language landscape)

| Paper | What they did | Simple version |
|---|---|---|
| LoRA (Hu 2021) | Proposed attaching small adapters to a frozen big model | "Cheap way to teach a model new tricks" |
| AdaLoRA (Zhang 2023) | Made the adapter *size* different per layer | "Some notes are bigger, some smaller" |
| IST (Yao 2024) | Picked *which layers* to attach to, per task | "Don't stick notes on useless layers" |
| Task Arithmetic (Ilharco 2023) | Showed fine-tuned models can be added/subtracted | "Task knowledge lives in specific weight directions" |

### The gap you fill (in one sentence)

**Nobody has directly tested whether the list of "useful layers" for one task also works for another task.** IST assumes you need a fresh search for every task. You're going to test that assumption.

---

## What you'll actually do (the work, in plain words)

1. **Pick 3 small models** that fit your 4GB GPU: GPT-2 medium, Pythia-410M, TinyLlama-1.1B.
2. **Pick 6 common NLP tasks**: sentiment, topic classification, NLI, paraphrase, QA-style, movie reviews.
3. **For each task, find the "best" layers.** You do this by attaching a LoRA adapter to *just one layer at a time*, training briefly, and seeing how well it performs. Repeat for every layer. The layers where single-layer LoRA does best = most important layers for that task.
4. **Compare rankings across tasks.** If sentiment's top layers are also topic's top layers → transfer works. You measure this with a simple number called Spearman correlation (0 = no match, 1 = perfect match).
5. **Build a "universal" placement.** Combine all the task rankings into one consensus list. Test: if you always use this fixed list, how much worse are you than searching per-task? If the gap is small, you have a useful paper.
6. **Write the paper.** 10 pages, Scopus journal format.

That's it. 6 steps. The first 5 are mostly running scripts and letting the GPU cook overnight.

---

## Topics you need to study — ordered by urgency

### Level 1 — The basics (skip if already comfortable)

| Topic | Why | Minimum to know | Time |
|---|---|---|---|
| PyTorch fundamentals | All your code uses it | `nn.Module`, `optimizer.step()`, `loss.backward()`, train vs eval mode | 1 day refresher |
| HuggingFace Transformers | How you'll load models | `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer` | 0.5 day |
| Basic Linux shell + Git | Running scripts, saving code | `ssh`, `python train.py`, `git commit -m` | Already know |

**Good free resources:** Sebastian Raschka's *Build a Large Language Model from Scratch* (you already have notes on this in the wiki — see [[gpt2-from-scratch]]). HuggingFace's free course at huggingface.co/learn.

### Level 2 — Core LLM concepts you must own

| Topic | What to understand | Your wiki page |
|---|---|---|
| Transformer layers | What a layer is, how attention + FFN compose one | [[transformer-architecture]] · [[multi-head-attention]] |
| Decoder-only architecture | GPT-style structure | [[decoder-only-architecture]] |
| Fine-tuning | What "fine-tuning" means vs training from scratch | [[fine-tuning]] |
| Tokenization + embeddings | How text becomes numbers | [[tokenization]] · [[embeddings]] |

You're already strong here — the wiki has dense notes. Just re-read if rusty.

### Level 3 — LoRA (the technique at the heart of your paper)

**Read these 3 papers in order, then stop.** Don't read all 50 LoRA variants.

1. **LoRA original** (Hu et al., 2021) — [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685). Read: abstract, Section 3 (method), Figure 1. Skip the rest first pass. Understand: *LoRA adds low-rank matrices A and B alongside frozen weights; trains only A and B.*
2. **AdaLoRA** (Zhang 2023) — [arxiv.org/abs/2303.10512](https://arxiv.org/abs/2303.10512). Read: abstract, Section 3, Section 4.1. Understand: *it adapts rank per layer, not placement.* This is a baseline.
3. **IST / Layer-wise Importance Matters** (Yao 2024) — [aclanthology.org/2024.findings-emnlp.109.pdf](https://aclanthology.org/2024.findings-emnlp.109.pdf). Read: abstract, method section, results table. This is your MAIN baseline — you must understand it deeply.

**Practical tool:** HuggingFace PEFT library. Spend half a day on the [PEFT LoRA tutorial](https://huggingface.co/docs/peft/en/task_guides/lora_based_methods). You'll write ~30 lines of PEFT code total for the whole paper.

### Level 4 — Measurement and statistics (light touch)

| Topic | What you need | Resource |
|---|---|---|
| Spearman rank correlation | What it measures + how to compute it (`scipy.stats.spearmanr`) | 1 hour: Wikipedia + scipy docs |
| Train / val / test splits | Why they exist + never touch test until the end | Already know |
| Random seeds | Why 3 seeds, how to set them in PyTorch | 1 hour: read any reproducibility guide |
| T-test / p-value | Just enough to report "p < 0.05" correctly | 2 hours: Khan Academy or StatQuest video |

Don't go deep into stats. You need basic rigor, not a statistician's toolkit.

### Level 5 — Writing the paper

| Topic | What to learn | Resource |
|---|---|---|
| LaTeX basics | Enough to use an IEEE Access template | Overleaf 30-minute tutorial |
| Paper structure | Abstract → Intro → Related → Method → Experiments → Results → Discussion → Conclusion | Read 2 Scopus papers in your area; notice the pattern |
| Good ML writing | How to write clearly without fluff | Zinsser's *On Writing Well* (1 chapter) or Strunk & White |

**Essential habit:** before you start writing, re-read 3 papers from your target venue (IEEE Access) to learn the voice and structure. This pays off enormously.

---

## Learning roadmap — when to study what

The best approach is **study while you set up, not before.** Don't try to "finish learning" before starting. Learn what you need, when you need it.

| Week | Experiment work | Study alongside |
|---|---|---|
| **Week 0 (prep)** | — | LoRA paper (Hu 2021). PyTorch refresher if rusty. |
| **Week 1** | Set up pipeline, run first LOLO test | HuggingFace PEFT tutorial. Read IST paper fully. |
| **Week 2** | LOLO on GPT-2 medium | Spearman correlation basics. Reproducibility / seeds. |
| **Week 3** | LOLO on Pythia + TinyLlama | AdaLoRA paper (for your baselines section). |
| **Week 4** | Cross-apply + universal placement | Start reading 2 IEEE Access papers in your area for writing style. |
| **Week 5** | Run baselines | T-test / p-value basics. |
| **Week 6** | Start writing — Introduction + Related Work + Method | LaTeX + Overleaf if new to you. |
| **Week 7** | Finish writing — Results + Discussion + Conclusion | — |
| **Week 8** | Polish, format, submit | — |

You are *not* expected to know everything on day 1. You'll learn roughly 30% of what you need each week, at the moment you need it.

---

## Fears you might have — honest answers

**"I don't fully understand LoRA yet."**
You don't need to. You need to understand: *LoRA adds A and B matrices; HuggingFace PEFT handles the math; you choose which layers to attach it to.* Everything else is implementation detail the library handles.

**"350 GPU-hours sounds like a lot."**
It's not. That's ~2 weeks of overnight runs. You sleep, the GPU works. You're not babysitting it. Each individual run is 10–15 minutes. Write a script, queue them up, check in the morning.

**"What if I get stuck on a bug?"**
You will. Every researcher does. Rubber-duck debugging works. Stack Overflow works. When truly stuck, ask Claude / GPT-5 in a specific way: paste the full error + relevant 30 lines of code. Do not ask vague questions like "my training isn't working."

**"Does this count as an SCI paper or only a Scopus paper?"**
Both. All 5 target venues in the technical plan (IEEE Access, Applied Sciences, Neural Computing and Applications, Expert Systems with Applications, Neurocomputing) are indexed in *both* Scopus and SCIE (the modern name for SCI). If your paper gets accepted at any of them, it counts as both — so UGC points, PhD applications, and career credit work either way. You do not need a separate strategy for "SCI journals vs Scopus journals."

**"What if my results are boring (no transfer / too much transfer)?"**
Both outcomes publish. A boring positive result = "universal placement works, here's the recipe." A boring negative result = "contrary to implicit assumptions, placement does not transfer; per-task search is necessary." IEEE Access-tier journals accept both framings cleanly. The only outcome that doesn't publish is "I didn't run the experiments carefully" — so just be careful.

**"I'm alone. Nobody is checking my methodology."**
True. Compensate by: (a) following the experimental protocol in [[research-p3-sparse-lora]] exactly, (b) asking an LLM to review each section of your paper before submission, (c) posting a draft on ML Twitter / r/MachineLearning if you want a free pre-review.

**"Can I really do this in 2 months alone?"**
Yes, if you:
- Stick to the 8-week plan
- Don't add scope (no extra models, no extra tasks, no FFN placement)
- Let the fail-safe (negative result) stay on the table
- Write ~1 hour per day, not in a panic at the end

**"What if I need to extend beyond 2 months?"**
Budget 3 months in your head, 2 months in your schedule. If Week 8 slips to Week 10, fine. Submit in month 3.

---

## A note on difficulty

This is a real research paper. It will stretch you. That's the point of a master's. But it is *not* a PhD thesis, and you should not treat it like one. Your job is to answer one narrow, specific question (does layer placement transfer?) carefully, report the results honestly, and move on. You are not trying to revolutionize PEFT. You're adding one honest data point. That is exactly what Scopus journals want.

When you feel overwhelmed, come back to this file. The full technical plan is in [[research-p3-sparse-lora]] when you need the specifics.

---

## Related

- [[research-p3-sparse-lora]] — the technical research plan (reference when you need specifics)
- [[fine-tuning]]
- [[large-language-models]]
- [[gpt2-from-scratch]]
- [[transformer-architecture]]
- [[multi-head-attention]]
- [[tokenization]]
- [[embeddings]]
- [[decoder-only-architecture]]
