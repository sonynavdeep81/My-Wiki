---
title: GPT-2 From Scratch (Notebook)
type: source
tags: [gpt2, pytorch, implementation, training, inference, fine-tuning, sampling]
sources: 1
updated: 2026-04-18
---

## GPT-2 From Scratch (Notebook)

**Summary**: A complete PyTorch implementation of GPT-2 built from scratch, covering architecture, training on a small text corpus, inference with temperature/top-k/multinomial sampling, loading OpenAI pretrained weights, and classification fine-tuning.

## Notebook Structure (72 cells)

1. **Import Modules** — tiktoken, torch, numpy, pandas
2. **Read text file** — `the-verdict.txt` (from rasbt/LLMs-from-scratch)
3. **QKV Bias** — `False` for scratch training; `True` required for loading OpenAI weights
4. **Create Data Batches** — `GPTDataset` (stride-based sliding window) + `create_dataloader`
5. **Train/Val Split** — 90/10 character-based split
6. **Model Architecture** — `LayerNorm`, `GELU`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `GPT2Model`
7. **Total Trainable Parameters** — ~162M
8. **Model Training** — AdamW (lr=0.0004, weight_decay=0.1), cross-entropy loss, CUDA-aware
9. **Saving/Loading** — `torch.save/load` with model + optimizer state_dict
10. **Model Inference**
    - **Temperature** — divides logits before softmax; controls distribution sharpness
    - **torch.multinomial()** — samples from probability distribution (not argmax); full pipeline: `logits → /T → softmax → probs → multinomial → token index`
    - **Top-k Sampling** — restricts candidate pool to top-k tokens before softmax; always used with temperature
11. **Loading GPT-2 Weights from OpenAI** — `load_weights_into_gpt2()` maps OpenAI checkpoint format
12. **What is Fine-Tuning?** — instruction vs classification overview with diagrams; PEFT/LoRA/QLoRA tradeoffs
13. **Implementing Classification Fine-Tuning** — SMS spam demo (UCI dataset, balanced, 70/10/20 split, binary labels spam=1/ham=0)

## Config

```python
GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'emb_dim': 768,
    'context_length': 256,   # truncated from GPT-2's 1024 for this demo
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True  # True to load OpenAI weights; False for scratch training
}
```

## Key Implementation Details

### TransformerBlock (Pre-LN)
```python
# Attention sub-block
shortcut = x
x = self.ln1(x)      # normalize BEFORE attention
x = self.att(x)
x = self.dropout(x)
x = x + shortcut     # residual

# FFN sub-block
shortcut = x
x = self.ln2(x)      # normalize BEFORE FFN
x = self.ff(x)
x = self.dropout(x)
x = x + shortcut
```

### Causal Mask as register_buffer
```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
```
Registered as a non-trainable tensor — moves to GPU with `.to(device)` automatically.

### Weight Tying
`out_head.weight` shares the same tensor as `tok_emb.weight` (both assigned from OpenAI's `params['wte']`). Reduces parameters and stabilizes training. See [[weight-tying]].

### Loading OpenAI Weights
OpenAI stores Q, K, V concatenated in a single `c_attn` matrix:
```python
q_w, k_w, v_w = np.split(params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
assign(model.trf_blocks[b].att.W_query.weight, q_w.T)
```
Also: OpenAI uses `qkv_bias=True` in their checkpoint — must match at model creation time.

### Inference Pipeline (Top-k + Temperature + Multinomial)

Full pipeline per generation step: `logits → /T → softmax → probs → multinomial → token index`

```python
logits = model(token_ids)[:, -1, :]          # last token's logits only
top_values, _ = torch.topk(logits, k=top_k)
logits[logits < top_values[:, -1]] = -inf    # mask non-top-k to -∞
probs = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

- **Top-k**: restricts *which* tokens are candidates
- **Temperature**: controls *how confidently* to pick (T<1 sharper; T>1 flatter; T→0 greedy; T→∞ uniform)
- **Multinomial**: the actual random draw — unlike `argmax`, allows lower-prob tokens occasionally


## Fine-Tuning

- **Instruction fine-tuning**: updates full model over long sequences/diverse tasks; high compute; use PEFT (LoRA/QLoRA) to reduce cost
- **Classification fine-tuning**: replaces output head; lower compute; SMS spam demo (UCI dataset, balanced, 70/10/20 split, binary labels spam=1/ham=0)

## New Concepts

- [[gpt2-from-scratch]]
- [[decoding-strategies]]
- [[fine-tuning]]
- [[weight-tying]]

## Entities

- [[gpt-family]]

## Related

- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[layer-normalization]]
- [[feed-forward-network]]
- [[kv-caching]]
