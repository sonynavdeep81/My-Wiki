---
title: Research B3 — Prompt Sensitivity in Zero-Shot Classification
type: query
tags: [research, prompt-engineering, zero-shot, sensitivity, gpt2, llama, scopus]
sources: 0
updated: 2026-04-17
---

## Research B3 — Prompt Sensitivity in Zero-Shot Classification

**Summary**: A complete implementation guide for studying how much prompt phrasing affects zero-shot classification accuracy — a Scopus journal topic requiring no model training, only inference.

---

## Core Idea

Zero-shot classification = asking a model to label text without any training examples.

The same question can be phrased many ways:
```
P1: "Is this review positive or negative? {text} Answer:"
P2: "Does this text express a good or bad opinion? {text} Answer:"
P3: "Sentiment of the following: {text} The sentiment is"
```

All three mean the same thing. The model gives **different accuracy** for each. B3 quantifies how much this matters and identifies which phrasing patterns consistently win.

---

## Understanding the Metrics

### Accuracy
- Out of 100 examples, how many did the model label correctly
- Higher = better; random baseline for 2-class task = 50%
- Measures whether a specific prompt is actually useful

### Sensitivity Score
```
sensitivity = best_prompt_accuracy − worst_prompt_accuracy
```

| Sensitivity | Meaning |
|---|---|
| < 10% | Model stable — wording barely matters |
| 10–20% | Moderate — prompt choice matters |
| > 20% | Unstable — wrong phrasing seriously hurts performance |

**High sensitivity = interesting finding.** It means the model is pattern-matching prompt surface rather than understanding the task. Like a student who only answers correctly when the question uses textbook language — not because they understand the topic, but because they recognise the phrasing.

### Why High Values Matter for the Paper

| Finding | Implication |
|---|---|
| High sensitivity on task X | Task X is fragile for zero-shot — needs careful prompting |
| Sensitivity varies by task | Some tasks are more prompt-dependent than others |
| LLaMA less sensitive than GPT-2 | Scale reduces prompt fragility — empirical evidence |
| Fill-in-the-blank always wins | Pattern aligns with pretraining objective — explains *why* |

---

## Novel Contributions

1. **Sensitivity score per task** — quantifies the magnitude of prompt variance (not previously measured systematically)
2. **Prompt pattern taxonomy** — question / instruction / fill-in-the-blank / continuation, with average accuracy per type
3. **Practical design guidelines** — derived from winning patterns; tells practitioners how to write prompts

---

## Step-by-Step Implementation

### Phase 1 — Setup (Week 1)

**Install dependencies:**
```bash
pip install transformers datasets torch pandas scikit-learn
```

**Tasks and datasets (all free on HuggingFace):**

| Task | Dataset | Labels |
|---|---|---|
| Sentiment | SST-2 | positive / negative |
| Topic classification | AG News | world / sports / business / tech |
| Spam detection | SMS Spam | spam / ham |
| Textual entailment | SNLI | entail / contradict / neutral |

**Write 10 prompt variants per task (40 total):**

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

### Phase 2 — Run Experiments (Week 2–3)

**Load models:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"  # swap for "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

**Zero-shot classification logic:**
```python
def zero_shot_classify(model, tokenizer, prompt, label_words):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Get logits for the next token position
    next_token_logits = outputs.logits[0, -1, :]
    # Compare probabilities of each label word
    scores = {
        word: next_token_logits[tokenizer.encode(word)[0]].item()
        for word in label_words
    }
    return max(scores, key=scores.get)
```

**Run all combinations:**
```python
results = []
for model_name in ["gpt2", "llama"]:
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

### Phase 3 — Analysis (Week 4)

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

**Expected output — Table 1:**

| Task | GPT-2 sensitivity | LLaMA sensitivity |
|---|---|---|
| Sentiment | ~18% | ~11% |
| Topic | ~25% | ~16% |
| Spam | ~9% | ~6% |
| Entailment | ~31% | ~22% |

**Build pattern taxonomy:**

Label each of your 40 prompts by type:
- **Question form** — "Is this positive or negative?"
- **Instruction form** — "Classify the sentiment of:"
- **Fill-in-the-blank** — "This review is ___"
- **Continuation form** — "Review: [text] The sentiment is"

Calculate average accuracy per type across all tasks.

**Expected output — Table 2:**

| Pattern type | Avg accuracy |
|---|---|
| Fill-in-the-blank | ~72% |
| Continuation form | ~68% |
| Instruction form | ~61% |
| Question form | ~59% |

**Identify winning patterns:**
Look at top-3 prompts per task — what do they share? Word order, label words in prompt, sentence structure. These become your guidelines.

---

### Phase 4 — Write the Paper (Week 5–6)

**Paper structure:**

| Section | Content |
|---|---|
| Introduction | Why prompt sensitivity matters for zero-shot NLP |
| Related Work | Existing prompt engineering / calibration papers |
| Experimental Setup | Tasks, datasets, models, 40 prompt variants |
| Results | Table 1 (sensitivity), Table 2 (taxonomy) |
| Discussion | Winning pattern guidelines + why fill-in-the-blank wins |
| Conclusion | Summary + limitations + future work |

**Target length:** 8–12 pages

---

## Total Timeline

| Week | Activity |
|---|---|
| 1 | Setup environment; write 40 prompt variants |
| 2–3 | Run all model × task × prompt combinations |
| 4 | Compute sensitivity scores; build taxonomy; extract guidelines |
| 5–6 | Write paper |

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
