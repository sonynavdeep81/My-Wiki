---
title: Fine-Tuning
type: concept
tags: [fine-tuning, instruction-tuning, classification, LoRA, PEFT, transfer-learning]
sources: 2
updated: 2026-04-30
verified_against: classification_fine_tuning, 2026-04-30
confidence: high
---

## Fine-Tuning

**Summary**: Additional training of a pre-trained (foundational) LLM on a smaller, domain-specific dataset to adapt it for a particular task or behavior.

## Two Main Types

### Instruction Fine-Tuning

Trains the model to follow natural language instructions across diverse tasks (e.g., "Summarize this:", "Translate to French:").

- Requires large, diverse instruction datasets
- Updates the **entire model** over long sequences
- High compute and memory demand
- Produces general-purpose chat/assistant models (e.g., InstructGPT, ChatGPT)

### Classification Fine-Tuning

Replaces the LM output head with a classification head (e.g., 2-class output for spam/ham). Trains on labeled examples for a specific task.

- Smaller dataset sufficient
- Lower compute — often only the head (and sometimes a few top layers) are trained
- Produces task-specific models

**Example**: SMS spam classification on UCI dataset
- Balanced dataset (equal spam/ham samples)
- Labels: spam=1, ham=0
- Split: 70% train / 10% val / 20% test

### Classification Head Replacement

```python
model.out_head = nn.Linear(emb_dim, num_classes, bias=True)  # bias=True required
```

`bias=True` is mandatory — `bias=False` silently kills convergence (zero gradient on untrained head).

### Freeze Strategy (GPT-2 Classification)

```python
# Step 1: freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 2: unfreeze head + final transformer block + final layer norm
for param in model.out_head.parameters():   param.requires_grad = True
for param in model.trf_blocks[-1].parameters(): param.requires_grad = True
for param in model.final_norm.parameters(): param.requires_grad = True
```

- Lower 11 blocks frozen — preserve general pre-trained representations
- Last block + norm unfrozen — adapts top-level features to classification task

### drop_rate for Fine-Tuning

Set `drop_rate=0` in the config when fine-tuning (was 0.1 for pretraining). See [[dropout-during-finetuning]].

### Classification Forward Pass

Uses **last token's logits** only (not all positions):
```python
logits = model(input_batch)[:, -1, :]   # (batch_size, num_classes)
loss = nn.functional.cross_entropy(logits, target_batch)
```

## PEFT: Parameter-Efficient Fine-Tuning

Instruction fine-tuning's high compute cost led to **PEFT** — freeze most model weights and train only a small set of additional parameters.

PEFT is a **broad family** of methods. Members include: Prefix Tuning, Prompt Tuning, Adapters, LoRA, QLoRA, DoRA, AdaLoRA, Sparse LoRA. LoRA and QLoRA are two prominent members, not the full family.

### LoRA (Low-Rank Adaptation)

Injects trainable low-rank matrices into attention layers alongside frozen original weights:
```
W_updated = W_frozen + A × B   (where A: d×r, B: r×d, r << d)
```
Only A and B are trained — a tiny fraction of total parameters.

### QLoRA

Combines LoRA with **4-bit quantization** of the frozen base model weights. Further reduces memory usage, enabling fine-tuning of large models on consumer GPUs.

| Method | Trainable Params | Memory | Use Case |
|---|---|---|---|
| Full fine-tuning | 100% | High | Instruction tuning with resources |
| LoRA | ~0.1–1% | Medium | Instruction or classification |
| QLoRA | ~0.1–1% | Low | Large models on limited hardware |

PEFT is most valuable for instruction fine-tuning but can also be applied to classification when the base model is large.

## Checkpoint Saving & Resume Training

Save the best model whenever validation accuracy improves:

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

To resume: reinitialize model architecture + freeze strategy first, then load weights:

```python
model = GPT2Model(GPT_CONFIG_124M)
# re-apply freeze + head replacement...
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs_seen = checkpoint['epochs_seen']
for epoch in range(epochs_seen[-1], epochs_seen[-1] + 5):
    ...
```

- Architecture must be rebuilt before loading state dict — weights load into structure, not the other way round
- Resume with lower lr (e.g. `1e-5` vs initial `5e-5`) when continuing from a good checkpoint

## Inference (Classification)

```python
LABEL_MAP = {0: 'ham', 1: 'spam'}

padded = pad_tokens(tokenizer.encode(text), train_dataset.max_tokens)
padded = torch.tensor(padded).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    outputs = model(padded).squeeze(0)

print(LABEL_MAP[torch.argmax(outputs[-1, :]).item()])
```

- Pad/truncate to the same `max_tokens` used during training — model expects fixed-length input
- `outputs[-1, :]` picks last token logits, consistent with training forward pass

## max_tokens Consistency Rule

When building datasets for classification fine-tuning:
- Compute `max_tokens` from the **training set only**: `max(len(encoded) for encoded in train_encoded)`
- Pass the **same value** explicitly to validation and test datasets
- Mismatch causes shape errors or garbage output — model was trained on fixed-length inputs

```python
train_dataset = SpamDataset("train.csv", tokenizer)         # computes max_tokens
val_dataset   = SpamDataset("val.csv",   tokenizer, max_tokens=train_dataset.max_tokens)
test_dataset  = SpamDataset("test.csv",  tokenizer, max_tokens=train_dataset.max_tokens)
```

## Related

- [[large-language-models]]
- [[scaling-laws]]
- [[gpt2-from-scratch]]
- [[decoder-only-architecture]]
- [[llama]]
- [[dropout-during-finetuning]]
- [[classification-finetuning-strategy]]
- [[spam-dataset-implementation]]
- [[requires-grad-vs-no-grad]]
- [[why-concept-pages]]
