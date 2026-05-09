---
title: Instruction Fine-Tuning (Notebook)
type: source
tags: [instruction-tuning, alpaca, collate, training, generate, peft, lora]
updated: 2026-05-09
---

## Instruction Fine-Tuning (Notebook)

**Summary**: Colab notebook implementing instruction fine-tuning of GPT-2 355M on the Alpaca-style instruction-data.json dataset, covering data pipeline, collate padding, training loop, generation, and LLM-as-judge evaluation.

## Key Points

- Fine-tuning = additional training of pretrained model on smaller domain-specific dataset
- Pretrained models fail at instruction-following: they continue text instead of responding
- PEFT family: LoRA (low-rank adapters) and QLoRA (LoRA + 4-bit quantization) — briefly mentioned, not implemented here

## Dataset

- Source: `instruction-data.json` (~1100 entries), list of `{instruction, input, output}` dicts
- Split: 85% train / 10% test / 5% val
- Prompt template: Alpaca format (must match exactly at inference)

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:        ← omitted if empty
{input}

### Response:
{output}
```

## InstructionDataset

- Encodes full text (instruction + input + response) via tiktoken GPT-2 tokenizer
- Truncates to `context_length=1024` — risk: may cut mid-response; recommended fix: filter long samples

## custom_collate

- Pads each batch dynamically to `max(len)+1` using `pad_token_id=50256`
- Targets = inputs shifted right by 1
- Masks only padding tokens beyond first EOS with `ignore_index=-100`
- **Does NOT mask instruction tokens** — instruction tokens contribute to loss

## Training

| Param | Value |
|-------|-------|
| Model | GPT-2 355M (pretrained weights) |
| Optimizer | AdamW |
| lr | 0.00005 |
| weight_decay | 0.1 |
| batch_size | 4 |
| epochs | 2 (more causes val/train divergence) |
| drop_rate | 0 |
| Log interval | every 50 batches |

- Two loss trackers: `running_train/val_batch_losses` (every 50 batches) + `train/val_losses` (per epoch)
- Two plots: batch-level intermediate + epoch-level final

## Generate Function

- top_k=1, temperature=1 for deterministic evaluation (greedy — no creativity, exact answer)
- top_k filtering → softmax → multinomial sampling
- model.eval() + torch.no_grad() inside loop
- **EOS stop token** — stops generation when token 50256 (`<|endoftext|>`) is produced:
  ```python
  if next_token_id == 50256:
      break
  ```
  Without this, model continues hallucinating new prompts after the response ends. Token 50256 appearing is correct behavior — the model learned to signal "done" via EOS.

## Evaluation

- Saves all test responses to `instruction-data-with-response.json`
- LLM-as-judge using Groq API + `llama-3.1-8b-instant`
- API key via `getpass` (Colab-style — prompts user at runtime)
- Judge prompt: instruction + optional input + expected response + model response → score 0–100 + reason
- Scores saved to `evaluation_results.json`, downloaded via `files.download()`
- Evaluated on first 10 entries of test set (`test_data_response[:10]`)

## Backlinks

- [[instruction-fine-tuning]]
- [[decoding-strategies]]
- [[lora]]
- [[fine-tuning]]
- [[cross-entropy-loss]]
- [[dropout]]
