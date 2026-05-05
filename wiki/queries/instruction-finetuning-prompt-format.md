---
title: Instruction Fine-Tuning — Prompt Format (Alpaca and Others)
type: query
tags: [fine-tuning, instruction-tuning, alpaca, prompt-format, template]
sources: 1
updated: 2026-04-30
---

## Instruction Fine-Tuning — Prompt Format (Alpaca and Others)

**Summary**: No universal format standard exists; Alpaca popularized one common template but the only hard rule is training/inference format must match.

## What Alpaca Is

Stanford Alpaca = fine-tuned LLaMA trained on 52K GPT-3-generated instruction-following examples. Introduced a widely copied prompt template.

**Alpaca template:**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}    ← optional; omitted if no extra context

### Response:
{response}
```

## Is This Format Required?

No. Different models use different formats:

| Model/System | Format |
|---|---|
| Stanford Alpaca | `### Instruction / ### Input / ### Response` |
| Vicuna | `USER: ... ASSISTANT: ...` |
| ChatML (OpenAI) | `<\|im_start\|>user\n...<\|im_end\|>` |
| LLaMA-3 | `<\|user\|>...<\|assistant\|>` |

## The One Hard Rule

**Training format = inference format.** Whatever template wraps examples during fine-tuning must be used identically at inference. Mismatch degrades output quality because the model learned to expect specific delimiter tokens as task boundaries.

## Related

- [[fine-tuning]]
- [[instruction-finetuning-data-format]]
- [[stanford-alpaca]]
- [[llama]]
- [[large-language-models]]
- [[large-language-models]]
- [[llama]]
