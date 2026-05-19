---
title: Instruction Fine-Tuning (Notebook)
type: source
tags: [instruction-tuning, alpaca, collate, training, generate, peft, lora, groq]
updated: 2026-05-19
---

## Instruction Fine-Tuning (Notebook)

**Summary**: Colab notebook implementing instruction fine-tuning of GPT-2 355M on the Alpaca-style instruction-data.json dataset, covering data pipeline, collate padding, training loop, two-version generate() progression, and LLM-as-judge evaluation via Groq.

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

Two versions shown in the notebook (pedagogical progression):

**V1 — Partial fixes applied (still buggy):**
- `logits = model(token_ids[:, -context_size:])` — sliding window fix present
- `min_val = top_values[:, -1].unsqueeze(-1)` — broadcasting fix present
- **No EOS stop** — model hallucinates new prompts after response ends
- **No `return`** — only `print(tokens)`; caller cannot use the output
- `model_device` referenced from outer scope (not defined inside function)

**V2 — Fully fixed:**
```python
def generate(token_ids, max_length, top_k, temperature, context_size):
  model_device = next(model.parameters()).device   # defined inside function
  token_ids = token_ids.to(model_device)
  model.eval()
  with torch.no_grad():
    for i in range(max_length):
      logits = model(token_ids[:, -context_size:])  # sliding window
      logits = logits[:, -1, :]
      top_values, top_indices = torch.topk(logits, k=top_k)
      min_val = top_values[:, -1].unsqueeze(-1)     # correct broadcast shape (batch,1)
      logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=logits.device), logits)
      scaled_logits = logits / temperature
      probs = torch.softmax(scaled_logits, dim=-1)
      next_token_id = torch.multinomial(probs, num_samples=1)
      if next_token_id.item() == 50256:             # .item() → plain int; EOS stop
        break
      token_ids = torch.cat([token_ids, next_token_id], dim=-1)
  tokens = tokenizer.decode(token_ids.squeeze(0).tolist())
  return tokens
```

Settings for deterministic evaluation: `top_k=1, temperature=1` (greedy — no creativity, exact answer)

## Evaluation

- Saves all test responses to `instruction-data-with-response.json`
- Response generation: iterates over **all** `test_data` entries; saves to `instruction-data-with-response.json`
- Safety guard: if model doesn't emit `### Response:`, sets `model_response = ""` and skips
- LLM-as-judge using **Groq API** + `llama-3.1-8b-instant` — judges **first 10** entries only
  - Groq = hardware/infrastructure (LPU chips); LLaMA = Meta's model weights — two separate things
  - Groq hosts Meta's model and serves it fast; no Meta public API exists
- API key via `getpass` (Colab-style — prompts user at runtime; hidden input)
- `time.sleep(0.5)` between judge calls to avoid Groq rate limit
- Judge prompt: instruction + optional input + expected response + model response → score 0–100 + reason
- Scores saved to `evaluation_results.json`, downloaded via `files.download()`
- Notebook includes inline Groq vs LLaMA explanation block

## Backlinks

- [[instruction-fine-tuning]]
- [[decoding-strategies]]
- [[lora]]
- [[fine-tuning]]
- [[cross-entropy-loss]]
- [[dropout]]
