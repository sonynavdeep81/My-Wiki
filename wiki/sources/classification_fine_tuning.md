---
title: Classification Fine-Tuning (Python)
type: source
tags: [fine-tuning, classification, spam, pytorch, gpt2, freeze, SpamDataset]
sources: 1
updated: 2026-04-30
---

## Classification Fine-Tuning (Python)

**Summary**: Python script implementing GPT-2 classification fine-tuning for SMS spam detection — covers dataset preparation, freeze strategy, head replacement, and training loop with accuracy evaluation.

**Source file**: `raw/classification_fine_tuning.py`

## Script Structure

1. **Architecture** — Reuses `GPT2Model`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `MultiHeadAttention` (identical to `gpt2_decoder.py`)
2. **Config** — GPT_CONFIG_124M with `drop_rate=0` (changed from 0.1; dropout off for fine-tuning)
3. **Load OpenAI Weights** — `assign` + `load_weights_into_gpt2` (same as decoder script)
4. **Fine-Tuning Overview** — instruction vs classification comparison; PEFT/LoRA/QLoRA tradeoffs
5. **Dataset Preparation** — `create_balanced_dataset`, `random_split` (70/10/20), `SpamDataset`
6. **DataLoaders** — batch_size=8; train: shuffle=True, drop_last=True; val/test: shuffle=False
7. **Head Replacement + Freeze** — replace out_head; freeze all; unfreeze head + last block + final norm
8. **Training Loop** — `cal_batch_loss`, `cal_loader_loss`, `cal_accuracy_loader`, `plot_losses`

## Config (Classification)

```python
GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'emb_dim': 768,
    'context_length': 256,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0,      # 0 for fine-tuning (was 0.1 for pretraining)
    'qkv_bias': True     # must match loaded OpenAI weights
}
```

## SpamDataset

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_tokens=None, pad_token_id=50256):
        # tokenize all texts
        # if max_tokens is None: max_tokens = max(len(enc) for enc in encoded_texts)
        # truncate to max_tokens
        # pad shorter sequences to max_tokens with pad_token_id
```

- `max_tokens=None` on train → auto-set to longest sequence in train set
- Same value passed explicitly to val/test datasets
- `assert train_dataset.max_tokens <= context_length` — prevents positional embedding crash

## Head Replacement & Freeze Strategy

```python
# Replace LM head with 2-class classifier
model.out_head = nn.Linear(cfg['emb_dim'], 2, bias=True)   # bias=True required

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze: output head + last transformer block + final layer norm
for param in model.out_head.parameters():
    param.requires_grad = True
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

- `bias=True` on out_head — required for convergence; `bias=False` silently kills learning
- Unfreeze last block: preserves general features in lower layers; fine-tunes task-specific representations at top

## Classification Forward Pass

Classification uses **last token's logits** only:
```python
logits = model(input_batch)[:, -1, :]   # shape: (batch_size, num_classes)
loss = nn.functional.cross_entropy(logits, target_batch)
```

Different from LM loss which uses all token positions.

## Dataset

- UCI SMS Spam Collection
- Labels: spam=1, ham=0
- Balanced via `create_balanced_dataset` (equal class counts)
- Split: 70% train / 10% val / 20% test (shuffled via `random_split`)
- pad_token_id=50256 (`<|endoftext|>`)

## New Concepts

- [[fine-tuning]]
- [[dropout]]

## Related

- [[gpt2-from-scratch]]
- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[weight-tying]]
- [[optimizer]]
- [[requires-grad-vs-no-grad]]
- [[dropout-during-finetuning]]
- [[spam-dataset-implementation]]
- [[classification-finetuning-strategy]]
- [[train-val-test-split]]
- [[dataloader-parameters]]
