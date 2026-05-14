---
title: Instruction Fine-Tuning — Prompt Format (Alpaca and Others)
type: query
tags: [fine-tuning, instruction-tuning, alpaca, prompt-format, template]
sources: 1
updated: 2026-05-14
---

## Instruction Fine-Tuning — Prompt Format (Alpaca and Others)

**Summary**: There is no universal prompt format standard. Alpaca popularized one common template, but different models use different formats. The only hard rule: training format must exactly match inference format.

---

## What Is a Prompt Format?

When training a model on instruction-response pairs, you cannot just concatenate "instruction + response" as raw text. The model needs clear **delimiters** — markers that tell it where the instruction ends and where the response begins. These delimiters are part of the prompt format (also called a prompt template).

During inference, you provide the instruction wrapped in the same template, up to the response delimiter. The model then generates the response.

---

## The Alpaca Format

Stanford Alpaca (2023) — a fine-tuned LLaMA model trained on 52,000 GPT-3-generated instruction examples — popularized this template:

```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

- `### Instruction:` — the task description
- `### Input:` — optional extra context (e.g., a paragraph to summarize). Omitted if not needed.
- `### Response:` — the model's answer goes here

This format became widely copied because Alpaca's code was open-sourced and easy to reproduce.

---

## Other Common Formats

Different models and systems developed their own formats:

| Model/System | Format |
|---|---|
| Stanford Alpaca | `### Instruction / ### Input / ### Response` |
| Vicuna | `USER: {instruction} ASSISTANT: {response}` |
| ChatML (OpenAI) | `<\|im_start\|>user\n{instruction}<\|im_end\|>\n<\|im_start\|>assistant\n{response}` |
| LLaMA-3 | `<\|user\|>{instruction}<\|assistant\|>{response}` |

None of these is "correct" — they are all conventions chosen by the teams that built the models.

---

## The One Hard Rule — Training Format Must Match Inference Format

Whatever template you use during fine-tuning must be used **identically** at inference time.

**Why:** During training, the model sees the delimiters thousands of times. It learns that `### Response:` means "now I should generate an answer." If you change the delimiter at inference — or omit it — the model loses this signal and produces garbage.

**Example:**

Training format:
```
### Instruction: Convert 45 km to meters.
### Response: 45 km is 45,000 meters.
```

Correct inference prompt:
```
### Instruction: Convert 45 km to meters.
### Response:
```
The model continues after `### Response:` with the answer.

Wrong inference prompt:
```
Convert 45 km to meters.
Answer:
```
The model never learned that `Answer:` means it should generate a response. Output quality degrades.

---

## Choosing a Format for Your Own Fine-Tuning

Any format works as long as:
1. It has clear delimiters between instruction and response
2. You use it consistently for every training example
3. You use the exact same format at inference time

For simplicity, the Alpaca format is a safe default — it is well-documented, widely understood, and many open-source datasets already use it.

---

## Related

- [[fine-tuning]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-data-pipeline]]
- [[stanford-alpaca]]
- [[llama]]
