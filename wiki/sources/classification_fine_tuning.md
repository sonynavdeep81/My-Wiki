---
title: Classification Fine-Tuning (Python)
type: source
tags: [fine-tuning, classification, spam, pytorch, gpt2, freeze, SpamDataset, checkpoint]
sources: 1
updated: 2026-05-16
---

## Classification Fine-Tuning (Python)

**Summary**: Python script implementing GPT-2 classification fine-tuning for SMS spam detection — covers dataset preparation, freeze strategy, head replacement, training loop with checkpoint saving/loading, and inference.

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
9. **Checkpoint Save/Load** — best model saved by val_accuracy; full resume training pattern
10. **Inference** — `pad_tokens()` + `LABEL_MAP` for real-text prediction

## Config (Classification)

```python
GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'emb_dim': 768,
    'context_length': 1024,  # matches GPT-2 positional embedding table size
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0,           # 0 for fine-tuning (was 0.1 for pretraining)
    'qkv_bias': True          # must match loaded OpenAI weights
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

## Checkpoint Saving

Best model saved to `best_model.pth` whenever `val_accuracy` improves during training:

```python
if val_accuracy > val_acc:
    val_acc = val_accuracy
    torch.save({
        'epoch': epoch + 1,
        'val_accuracy': val_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_seen': epochs_seen,
    }, 'best_model.pth')
```

- Saves both model and optimizer state — required for faithful resume
- Saves loss history so plots can continue from where they left off

## Checkpoint Loading & Resume Training

```python
checkpoint = torch.load("/content/best_model.pth")

# Reinitialize architecture + freeze strategy (must match training setup)
model = GPT2Model(GPT_CONFIG_124M)
for param in model.parameters():
    param.requires_grad = False
model.out_head = nn.Linear(GPT_CONFIG_124M['emb_dim'], num_classes, bias=True)
for param in model.trf_blocks[-1].parameters(): param.requires_grad = True
for param in model.final_norm.parameters():     param.requires_grad = True

# Load weights and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore tracking state
epochs_seen  = checkpoint['epochs_seen']
train_losses = checkpoint['train_losses']
val_losses   = checkpoint['val_losses']

# Continue training
for epoch in range(epochs_seen[-1], epochs_seen[-1] + 5):
    ...
```

- Architecture + freeze must be re-applied **before** `load_state_dict` — the state dict only holds weights, not structure
- Resume lr reduced to `1e-5` (was `5e-5`) — fine-tuning further from a good checkpoint

## Inference

```python
def pad_tokens(tokens, max_tokens, pad_token_id=50256):
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    else:
        tokens = tokens + [pad_token_id] * (max_tokens - len(tokens))
    return tokens

LABEL_MAP = {0: 'ham', 1: 'spam'}

encoded = tokenizer.encode(text)
padded  = pad_tokens(encoded, train_dataset.max_tokens)
padded  = torch.tensor(padded).unsqueeze(0).to(device)  # add batch dim

model.eval()
with torch.no_grad():
    outputs = model(padded).squeeze(0)          # remove batch dim

predicted = torch.argmax(outputs[-1, :]).item()
print(LABEL_MAP[predicted])
```

- `unsqueeze(0)` adds the batch dimension (model expects `(batch, seq_len)`)
- `outputs[-1, :]` picks the last token's logits — same as training forward pass
- `LABEL_MAP` reverses the integer encoding back to human-readable label

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
- [[why-save-optimizer-state]]
