---
title: Research B3 — Prompt Sensitivity in Zero-Shot Classification
type: query
tags: [research, prompt-engineering, zero-shot, sensitivity, gpt2, llama, mistral, phi3, scopus]
sources: 0
updated: 2026-04-17
---

## Research B3 — Prompt Sensitivity in Zero-Shot Classification

**Summary**: Full implementation guide for studying how prompt phrasing affects zero-shot classification accuracy across 4 models and 3+ domains — a Scopus journal target requiring no model training, only inference.

---

## Core Idea

Zero-shot classification = asking a model to label text without any training examples.

The same question can be phrased many ways:
```
P1: "Is this review positive or negative? {text} Answer:"
P2: "Does this text express a good or bad opinion? {text} Answer:"
P3: "Sentiment of the following: {text} The sentiment is"
```

All three mean the same thing. The model gives **different accuracy** for each. B3 quantifies how much this matters across models of different sizes and on domain-specific text (medical, legal, technical) — and identifies which phrasing patterns consistently win.

---

## Prior Work — What Already Exists

| Paper | Finding | Gap they leave |
|---|---|---|
| Zhao et al. 2021 (Calibrate Before Use) | Zero-shot accuracy is sensitive to prompt format; calibration with label priors fixes it | Tested only GPT-3; no domain-specific text |
| Lu et al. 2021 (Order Matters) | Few-shot performance varies dramatically with example order | Covers few-shot, not zero-shot |
| Webson & Pavlick 2022 (Are Prompts Logical?) | Prompts with misleading semantics still work; models don't reason about meaning | Covers instruction-tuned models only |

**The gap your paper fills:**
- None of these test *smaller open-source models* (GPT-2, Phi-3, Mistral)
- None compare sensitivity across *domain-specific text* (medical, legal, technical vs. general news)
- None combine sensitivity measurement with *calibration correction* to show how much calibration reduces fragility
- None produce a *cross-model sensitivity curve* showing how sensitivity shrinks as model size grows

---

## Novel Contributions

1. **Sensitivity score per task and domain** — quantifies the magnitude of prompt variance on both standard benchmarks and domain-specific corpora
2. **Prompt pattern taxonomy** — question / instruction / fill-in-the-blank / continuation, with average accuracy per type across models
3. **Cross-model sensitivity comparison** — sensitivity curve across 4 model sizes (774M → 3.8B → 7B → 8B); empirically shows whether scaling reduces fragility
4. **Practical design guidelines** — derived from winning patterns across domains; directly usable by NLP practitioners who cannot fine-tune

---

## Understanding the Metrics

### Accuracy
- Out of 100 examples, how many did the model label correctly
- Higher = better; random baseline for 2-class = 50%

### Sensitivity Score
```
sensitivity = best_prompt_accuracy − worst_prompt_accuracy
```

| Sensitivity | Meaning |
|---|---|
| < 10% | Stable — wording barely matters |
| 10–20% | Moderate — prompt choice matters |
| > 20% | Unstable — wrong phrasing seriously hurts |

**High sensitivity = interesting finding.** The model is pattern-matching prompt surface rather than understanding the task.

---

## Model Set

| Model | Size | Access | VRAM (4-bit) | Notes |
|---|---|---|---|---|
| GPT-2 Large | 774M | HuggingFace | ~1.5 GB | Full precision, any GPU |
| Phi-3 Mini | 3.8B | HuggingFace | ~2 GB | Microsoft; instruction-tuned; 4-bit |
| Mistral 7B | 7B | HuggingFace | ~4.5 GB | Strong 7B baseline; 4-bit |
| LLaMA 3 8B | 8B | HuggingFace (gated) | ~5 GB | Meta; requires access request |

All four run on a single 8 GB GPU (RTX 3070/4060) with 4-bit quantization.

---

## Step-by-Step Implementation

### Phase 1 — Setup (Week 1)

**Install dependencies:**
```bash
pip install transformers datasets torch pandas scikit-learn bitsandbytes accelerate
```

**Tasks and datasets:**

| Task | Dataset | Labels | Domain |
|---|---|---|---|
| Sentiment | SST-2 | positive / negative | General (movie reviews) |
| Topic classification | AG News | world / sports / business / tech | News |
| Spam detection | SMS Spam | spam / ham | General |
| Medical sentiment | MedQuAD subset / MTSamples | positive / negative | Medical |
| Legal tone | EUR-Lex subset | formal / informal | Legal |

**Write 10 prompt variants per task (50 total):**

Example for SST-2:
```
P1:  "Is this review positive or negative? {text} Answer:"
P2:  "Does this text express a good or bad opinion? {text} Answer:"
P3:  "Sentiment of the following: {text} The sentiment is"
P4:  "Review: {text} This review is"
P5:  "Classify as positive or negative: {text} Label:"
P6:  "What is the sentiment? {text} Sentiment:"
P7:  "The following review expresses a {text} opinion."
P8:  "Text: {text} Is the author happy or unhappy?"
P9:  "Analyze the tone of this review: {text} Tone:"
P10: "{text} The emotional tone of this text is"
```

All 10 must be semantically identical — same meaning, different wording.

---

### Phase 2 — Load Models with 4-bit Quantization (Week 1–2)

**GPT-2 Large (full precision):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
```

**Phi-3 / Mistral / LLaMA with 4-bit:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_name = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Meta-Llama-3-8B", "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Zero-shot classification function:**
```python
def zero_shot_classify(model, tokenizer, prompt, label_words):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]
    scores = {
        word: next_token_logits[tokenizer.encode(word)[0]].item()
        for word in label_words
    }
    return max(scores, key=scores.get)
```

---

### Phase 3 — Run Experiments (Week 2–3)

```python
results = []
for model_name, model, tokenizer in model_list:
    for task in tasks:
        for prompt_id, prompt_template in enumerate(prompts[task]):
            correct = 0
            for text, true_label in test_data[task]:
                prompt = prompt_template.replace("{text}", text)
                predicted = zero_shot_classify(model, tokenizer, prompt, label_words[task])
                correct += (predicted == true_label)
            accuracy = correct / len(test_data[task])
            results.append({
                "model": model_name, "task": task,
                "prompt_id": prompt_id, "accuracy": accuracy
            })
```

---

### Phase 4 — Analysis (Week 4)

**Compute sensitivity scores:**
```python
import pandas as pd

df = pd.DataFrame(results)
sensitivity = df.groupby(["model", "task"])["accuracy"].agg(
    sensitivity=lambda x: x.max() - x.min(),
    best_accuracy="max",
    worst_accuracy="min"
).reset_index()
```

**Expected output — Table 1 (sensitivity per model × task):**

| Task | GPT-2 Large | Phi-3 Mini | Mistral 7B | LLaMA 3 8B |
|---|---|---|---|---|
| Sentiment | ~18% | ~14% | ~11% | ~10% |
| Topic | ~25% | ~19% | ~15% | ~13% |
| Spam | ~9% | ~7% | ~6% | ~5% |
| Medical | ~28% | ~20% | ~16% | ~14% |
| Legal | ~30% | ~23% | ~18% | ~15% |

**Expected output — Table 2 (pattern taxonomy):**

| Pattern type | Avg accuracy |
|---|---|
| Fill-in-the-blank | ~72% |
| Continuation form | ~68% |
| Instruction form | ~61% |
| Question form | ~59% |

**Expected output — Table 3 (cross-model sensitivity curve):**

Plot sensitivity (y-axis) vs. model size in billions (x-axis) for each task — shows whether larger models are less sensitive. This is Contribution C3.

**Build pattern taxonomy:**

Label each of your 50 prompts by type. Calculate average accuracy per type across all tasks.

---

### Phase 5 — Write the Paper (Week 5–7)

**Paper structure:**

| Section | Content |
|---|---|
| Introduction | Why prompt sensitivity matters; gap in prior work on small open-source models |
| Related Work | Zhao 2021, Lu 2021, Webson & Pavlick 2022; what they measured vs. what you add |
| Experimental Setup | 5 tasks (2 domain-specific), 4 models, 50 prompt variants, 4-bit quantization |
| Results | Table 1 (sensitivity per model/task), Table 2 (taxonomy), Table 3 (cross-model curve) |
| Discussion | Winning pattern guidelines; domain-specific text is harder; scaling reduces fragility |
| Conclusion | Summary + limitations + future work (multilingual, instruction-tuned models) |

**Target length:** 8–12 pages

---

## Total Timeline

| Week | Activity |
|---|---|
| 1 | Setup; write 50 prompt variants; load all 4 models |
| 2–3 | Run all model × task × prompt combinations |
| 4 | Compute sensitivity scores; build taxonomy; cross-model curve |
| 5–7 | Write paper |

---

## Target Journals

- *Information Processing & Management* (Elsevier, Scopus)
- *IEEE Access* (IEEE, Scopus)

---

## Related

- [[research-topics-feasible]]
- [[large-language-models]]
- [[decoding-strategies]]
- [[gpt-family]]
- [[llama]]
- [[llm-evaluation-metrics]]
