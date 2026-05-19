---
title: Instruction Fine-Tuning
type: concept
tags: [fine-tuning, instruction-tuning, loss-masking, padding, collate, alpaca]
sources: 4
updated: 2026-05-08
verified_against: Raschka-LLM-2025, 2026-05-07
confidence: high
---

## Instruction Fine-Tuning

**Summary**: Fine-tune a pre-trained LLM to follow natural language instructions by training on (instruction, response) pairs with loss computed only on the response tokens.

## Key Distinction from Classification Fine-Tuning

| | Classification FT | Instruction FT |
|---|---|---|
| Output head | Replaced (Linear → num_classes) | Kept (original LM head) |
| Labels | Single integer per sample | Token IDs shifted left by 1 |
| Padding | Global max across dataset | Per-batch max (dynamic) |
| Loss masking | None needed | Instruction tokens + padding → -100 |
| Dataset size | Small (100s–1000s) | Large (10K–100K+) |

## Data Format

Raw entry (JSON):
```json
{"instruction": "...", "input": "...", "output": "..."}
```

Wrapped via prompt template (e.g. Alpaca):
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Full text = instruction_plus_input + `\n\n### Response:\n` + output.

## 5-Step Data Pipeline

1. **Format** — wrap (instruction, input, output) in chosen template
2. **Tokenize** — encode full text; truncate at `context_length` (truncation risk: may cut response)
3. **Pad** — within each batch, pad to `max_len+1` with `50256` (`<|endoftext|>`)
4. **Shift target** — `inputs = padded[:-1]`, `targets = padded[1:]`; extra +1 token ensures last target position exists
5. **Mask** — replace all padding `50256`s in targets (beyond first) with `-100`; first `50256` kept as real EOS prediction target

## Loss Masking

- `-100` = PyTorch `ignore_index` for `cross_entropy` — positions excluded from loss entirely
- **This implementation masks padding tokens only** — instruction tokens are NOT masked and DO contribute to loss
- First EOS token (50256) kept as real prediction target; all subsequent padding → -100

```python
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index  # keep first 50256, mask the rest
```

> Note: masking instruction tokens is a common recommendation, but Shi et al. 2024 ("Instruction Tuning With Loss Over Instructions," arXiv:2405.14394) showed that **not masking** instructions benefits LLM performance. Raschka 2025 does not mask instructions by default; masking left as an optional exercise. [contested]

## collate_fn / Dynamic Padding

Padding to global max wastes compute. `custom_collate` pads each batch to its own `max_len`:
- Sequences within a batch padded to `max(len(item) for item in batch) + 1`
- Passed to `DataLoader` via `collate_fn=custom_collate`
- All sequences in batch become equal length → stackable tensor

## Truncation Risk

If tokenized full text > `context_length`, slicing `[:context_length]` may cut into the response. Fix: filter out samples exceeding context_length before training.

## Training Hyperparameters (355M)

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| lr | 5e-5 |
| weight_decay | 0.1 |
| batch_size | 4 |
| epochs | 2 — more causes val/train divergence on small datasets |
| drop_rate | 0 |
| Log interval | every 50 batches |

Two loss trackers: `running_train/val_batch_losses` (batch-level) + `train/val_losses` (epoch-level). Two plots: intermediate batch chart + final epoch chart.

## EOS Stop Token and Context Window in generate()

**[notebook]** After fine-tuning, the model emits token 50256 (`<|endoftext|>`) to signal end of response — this is correct behavior (it was trained this way). Without stopping at EOS, the model continues hallucinating new prompts after the response.

Three bugs in V1; V2 fixes all of them:

| Bug | V1 | V2 |
|---|---|---|
| Context window | `model(token_ids[:, -context_size:])` present | same |
| Broadcasting | `min_val = top_values[:, -1].unsqueeze(-1)` present | same |
| EOS stop | missing — hallucinates after response | `if next_token_id.item() == 50256: break` |
| Return value | missing — only `print(tokens)` | `return tokens` |

Note: context_size and unsqueeze fixes were incorporated into V1 in the updated notebook (2026-05-19) — V1's only remaining bugs are missing EOS stop and missing `return`.

- Without EOS stop: model continues generating new prompts after response ends
- Without `return`: caller cannot use the generated text
- `model_device` also moved inside V2 function for self-containment

## LLM-as-Judge Evaluation

**[notebook]** Evaluation pipeline after fine-tuning:
1. Generate responses for all test entries → save to `instruction-data-with-response.json`
2. For each entry, send instruction + expected + model response to judge LLM
3. Judge (Llama 3.1 8B via Groq) returns score 0–100 + reason
4. Save all scores to `evaluation_results.json`

Judge prompt structure:
```
Instruction: {instruction}
Input: {input}          ← omitted if empty
Expected response: {output}
Model response: {model_response}

Score the model response from 0 to 100. Give a brief reason.
```

API key retrieved via `getpass` at runtime (Colab) or Kaggle Secrets (Kaggle).

## Strategies to Improve Performance

**[notebook]** After fine-tuning, if results are unsatisfactory:

1. **Tune hyperparameters** — adjust learning rate, batch size, or number of epochs
2. **More/better data** — increase training dataset size or diversify examples to cover broader topics and styles
3. **Prompt engineering** — experiment with different instruction formats to guide responses more effectively
4. **Larger base model** — a bigger pretrained model has greater capacity to capture complex patterns and generate more accurate responses
5. **PEFT** — use parameter-efficient fine-tuning techniques like [[lora]] instead of full fine-tuning

## Related

- [[fine-tuning]]
- [[stanford-alpaca]]
- [[instruction-finetuning-data-pipeline]]
- [[instruction-finetuning-data-format]]
- [[instruction-finetuning-prompt-format]]
- [[instruction-finetuning-training-mechanics]]
- [[instruction-finetuning-collate-padding-trick]]
- [[dropout-during-finetuning]]
- [[lora]]
