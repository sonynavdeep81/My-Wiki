---
title: Classification Fine-Tuning Workflow — Spam Detection
type: query
tags: [fine-tuning, classification, spam, dataset, freeze, forward-pass, gpt2]
sources: 2
updated: 2026-05-14
---

## Classification Fine-Tuning Workflow — Spam Detection

**Summary**: A complete walkthrough of how to take a raw spam/ham CSV dataset and fine-tune a pre-trained GPT-2 model to classify emails — covering dataset preparation, tokenization, padding, the freeze strategy, and the forward pass.

---

## Step 1: Balance the Dataset

The raw UCI SMS Spam dataset has far more ham than spam examples. If you train on an imbalanced dataset, the model learns to cheat — it can achieve ~85% accuracy by always predicting "ham" without learning anything meaningful.

The fix: truncate the majority class so both classes have the same number of samples.

```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])
```

After balancing, split into **70% train / 10% val / 20% test**.

---

## Step 2: Encode Labels as Integers

Machine learning models work with numbers, not strings. Convert:
- `"ham"` → `0`
- `"spam"` → `1`

This is done in the CSV before passing it to the dataset class.

---

## Step 3: Tokenize Each Email

Each email text is passed through the BPE tokenizer (tiktoken, `gpt2` encoding). The tokenizer converts a string like `"Win a free iPhone now!"` into a list of integer token IDs, for example `[5380, 257, 1479, 3275, 783, 0]`.

Every email becomes a different-length list of integers — emails have different word counts, so their token lists will not all be the same length.

---

## Step 4: Pad All Sequences to the Same Length

Neural networks require all inputs in a batch to be the same shape. So we need every token list to have the same length.

**How the length is determined:**

Compute `max_tokens` from the **training set only**:
```python
max_tokens = max(len(enc) for enc in train_encoded_texts)
```

This single value is then used for *all* datasets — train, val, and test. It is **not** recomputed per batch. Every sequence in every batch will be padded or truncated to this same fixed length.

> **Why not compute it per batch?** The model's positional embeddings are trained on fixed-length inputs. If the length changed per batch, the model would see different position patterns in training vs inference and produce garbage output.

**How padding works:**

Shorter sequences are padded at the end with token `50256` — which is the `<|endoftext|>` token, repurposed as a pad token. The model ignores these padding positions because only the **last token's output** is used for classification (see Step 6).

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_tokens=None, pad_token_id=50256):
        # tokenize all texts
        # if max_tokens is None: compute from this dataset's longest sequence
        # truncate any sequence longer than max_tokens
        # pad shorter sequences to max_tokens with pad_token_id
```

A safety assertion prevents positional embedding crashes:
```python
assert train_dataset.max_tokens <= context_length
```

**Passing max_tokens consistently:**
```python
train_dataset = SpamDataset("train.csv", tokenizer)                               # auto-computes max_tokens
val_dataset   = SpamDataset("val.csv",   tokenizer, max_tokens=train_dataset.max_tokens)
test_dataset  = SpamDataset("test.csv",  tokenizer, max_tokens=train_dataset.max_tokens)
```

---

## Step 5: Replace the Output Head and Apply the Freeze Strategy

The pre-trained GPT-2 model has a language model head (`out_head`) that outputs a probability distribution over all 50,257 vocabulary tokens — one per possible next token. For spam classification, we only need 2 outputs (ham, spam).

**Replace the head:**
```python
model.out_head = nn.Linear(cfg['emb_dim'], 2, bias=True)   # 768 → 2
```

> **Critical:** `bias=True` is mandatory. With `bias=False`, the head starts with zero weights AND zero bias, so it produces zero gradients and never learns anything — the model silently fails to train.

**The freeze strategy — what to freeze and what to train:**

The goal is to preserve the general language knowledge the model learned during pretraining (in the lower layers), while adapting the top-level representations to the classification task.

```python
# Step 1: Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 2: Unfreeze the head + last transformer block + final layer norm
for param in model.out_head.parameters():       param.requires_grad = True
for param in model.trf_blocks[-1].parameters(): param.requires_grad = True
for param in model.final_norm.parameters():     param.requires_grad = True
```

What this means in practice:

| Component | Status | Reason |
|---|---|---|
| Embedding layers | Frozen | Vocabulary representations are already good |
| Transformer blocks 0–10 | Frozen | General language features; don't overwrite |
| Transformer block 11 (last) | **Trained** | Adapts top-level features to classification |
| Final layer norm | **Trained** | Directly feeds into the new head |
| `out_head` (new 2-class head) | **Trained** | Brand new; must be trained from scratch |

Also set `drop_rate=0` in the config. Dropout is useful during pretraining (helps generalization), but during classification fine-tuning with a mostly frozen model, it introduces noise without benefit.

---

## Step 6: The Forward Pass — Last Token Only

This is the most important conceptual difference from pretraining.

During **pretraining**, the model predicts the next token at every position — it produces a loss over all token positions in a sequence.

During **classification fine-tuning**, we use only the **last token's output**:

```python
logits = model(input_batch)[:, -1, :]   # shape: (batch_size, 2)
loss = nn.functional.cross_entropy(logits, target_batch)
```

The model still runs its full forward pass over all tokens (it has to — each token attends to all previous tokens). But only the final token's logit vector is extracted. This vector has shape `(2,)` — one score for ham, one for spam. The class with the higher score is the prediction.

**Why the last token?** In a decoder-only transformer, each token attends to all tokens before it (causal masking). The last token therefore has attended to the entire email and summarizes the full context. It is the richest representation available.

---

## Summary: The Full Pipeline

```
Raw CSV (ham/spam)
    ↓ balance classes
    ↓ encode labels: ham=0, spam=1
    ↓ split 70/10/20
    ↓ tokenize each email → list of token IDs
    ↓ pad all sequences to max_tokens (from train set) with token 50256
    ↓ load pre-trained GPT-2 weights
    ↓ replace out_head: Linear(768, 2, bias=True)
    ↓ freeze all params; unfreeze head + last block + final norm
    ↓ forward pass: logits = model(batch)[:, -1, :]
    ↓ loss = cross_entropy(logits, labels)
    ↓ backward + AdamW step
```

---

## Related

- [[fine-tuning]]
- [[classification-finetuning-strategy]]
- [[spam-dataset-implementation]]
- [[dropout-during-finetuning]]
- [[requires-grad-vs-no-grad]]
- [[cross-entropy-loss]]
- [[decoder-only-architecture]]
- [[gpt2-from-scratch]]
