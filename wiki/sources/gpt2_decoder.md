---
title: GPT-2 Decoder — Architecture & Pretraining (Python)
type: source
tags: [gpt2, pytorch, implementation, training, inference, sampling]
sources: 1
updated: 2026-05-14
---

## GPT-2 Decoder — Architecture & Pretraining (Python)

**Summary**: Colab-exported Python script implementing GPT-2 from scratch in PyTorch — covering data batching, full architecture, pretraining on a small text corpus, inference with temperature/top-k/multinomial sampling, OpenAI pretrained weight loading, and a second inference pass on the loaded weights.

**Source file**: `raw/gpt2_decoder.py` (843 lines, Colab-exported, 2026-05-14 update)

## Script Structure

1. **Import Modules** — `tiktoken`, `torch`, `numpy`, `pandas`, `urllib`
2. **Read text file** — `the-verdict.txt` (from rasbt/LLMs-from-scratch)
3. **GPT_CONFIG_124M (first config — scratch training)** — `context_length=256`, `qkv_bias=True`
4. **QKV Bias section** — explains why `qkv_bias=False` is preferred for modern LLMs (LayerNorm β subsumes it; LLaMA/Mistral/Falcon all use False) and why `True` is required to load OpenAI weights
5. **Create Data Batches** — `GPTDataset` (stride-based sliding window) + `DataLoader` (train: shuffle+drop_last; val: neither)
6. **Sanity Check** — guards against `len(tokens) < context_length` empty-dataset failure
7. **Model Architecture** — `LayerNorm`, `GELU`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `GPT2Model`
8. **`__call__` vs `forward` note** — always use `model(x)` (handles hooks, autograd, internals); never `model.forward(x)`
9. **Weight Tying explanation** — without WT: 162M params; with WT: 124M params; per-side gradient flow described
10. **Dropout placement comment** — three dropout sites per transformer block (inside MHA on attention weights, after MHA output, after FFN output); all `drop_rate=0.1`; all auto-disabled at inference
11. **Total Trainable Parameters** — ~124M (after weight tying); 162M without
12. **Model Training** — `cal_batch_loss`, `cal_loader_loss`, AdamW, 20-epoch loop, plot losses, save checkpoint
13. **Saving and Loading model weights** — `torch.save/load` with `model_state_dict` + `optimizer_state_dict`
14. **Model Inference (post-scratch-training)** — top_k=15, temperature=1.4, max_length=25
15. **Loading GPT-2 Weights from OpenAI** — `download_and_load_gpt2`, `GPT_CONFIG_124M (second config — context_length=1024)`, `load_weights_into_gpt2`
16. **Model Inference (post-OpenAI-weight-loading)** — same code as #14 but top_k=25

## Configs (TWO blocks — note they differ)

```python
# First config — for scratch training (lines 38-46)
GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'emb_dim': 768,
    'context_length': 256,    # truncated for the demo dataset
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,         # True so OpenAI weights can be loaded later
}

# Second config — for OpenAI weight loading (lines 719-727)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,   # full GPT-2 124M context window
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,         # MUST match OpenAI checkpoint
}
```

## Key Implementation Details

### LayerNorm (custom)
```python
class LayerNorm(nn.Module):
    def __init__(self, cfg, eps=1e-5):
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(cfg['emb_dim']))   # γ
        self.shift = nn.Parameter(torch.zeros(cfg['emb_dim']))  # β
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std  = torch.std(x,  dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift
```
Hand-rolled (not `nn.LayerNorm`); uses `torch.std` (Bessel-corrected) rather than population variance. See [[layernorm-scale-shift-sharing]] for parameter-sharing details.

### GELU (exact formula)
```python
return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))
                                * (x + 0.044715 * torch.pow(x,3))))
```

### FeedForward
`Sequential(Linear(768, 3072), GELU(), Linear(3072, 768))`

### TransformerBlock (Pre-LN)
```python
# Attention sub-block
shortcut = x
x = self.ln1(x)      # normalize BEFORE attention
x = self.att(x)
x = self.dropout(x)  # dropout on attention output
x = x + shortcut     # residual

# FFN sub-block
shortcut = x
x = self.ln2(x)      # normalize BEFORE FFN
x = self.ff(x)
x = self.dropout(x)  # dropout on FFN output
x = x + shortcut
```

### MultiHeadAttention
- `W_query`, `W_key`, `W_value`, `W_out` all `nn.Linear(d_model, d_model, bias=qkv_bias)`
- `head_dim = d_model // n_heads = 64`
- Reshape (b, T, d_model) → (b, T, n_heads, head_dim) → transpose to (b, n_heads, T, head_dim)
- `att_scores = Q @ K.transpose(-1,-2) / sqrt(head_dim)`
- Mask via `att_scores.masked_fill(self.causal_mask[:T,:T], -inf)`
- Softmax → dropout on attention weights → `@ V` → transpose+reshape → `W_out`

### Causal Mask as register_buffer
```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(cfg['context_length'], cfg['context_length']), diagonal=1).bool())
```
Non-trainable, device-aware (moves with `.to(device)`). See [[causal-mask-bool]] and [[register-buffer]].

### Weight Tying
```python
self.out_head.weight = self.tok_emb.weight   # one tensor, two names
```
Without weight tying: 162M params (two independent 50257×768 matrices). With: 124M (shared). See [[weight-tying]] and [[gpt2-parameter-count]].

### GPTDataset (stride-based sliding window)
```python
for i in range(0, len(token_ids) - context_length, stride):
    input_tokens  = torch.tensor(token_ids[i : i+context_length])
    target_tokens = torch.tensor(token_ids[i+1 : i+context_length+1])
```
In this script `stride == context_length` → non-overlapping chunks.

### Sanity Check (new in this version)
```python
if len(train_tokens) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for training loader...")
```
Prevents the silent-empty-dataset failure when corpus is shorter than `context_length`.

### Loss Helpers
```python
def cal_batch_loss(input_batch, target_batch, model):
    logits = model(input_batch)                                    # (2,256) → (2,256,50257)
    loss = nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def cal_loader_loss(loader, model):
    total = 0.
    for input_batch, target_batch in loader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        total += cal_batch_loss(input_batch, target_batch, model).item()
    return total / len(loader)
```
Function names use shortened `cal_*` (vs Raschka's `calc_*`) — same logic.

### Training Loop (20 epochs)
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
for epoch in range(20):
    total_loss = 0.
    for input_batch, target_batch in train_loader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        optimizer.zero_grad()
        loss = cal_batch_loss(input_batch, target_batch, model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    model.eval()
    with torch.no_grad():
        val_loss = cal_loader_loss(val_loader, model)
    train_losses.append(train_loss); val_losses.append(val_loss); epochs_seen.append(epoch+1)
    model.train()
plot_losses(epochs_seen, train_losses, val_losses)
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'checkpoint.pth')
```

### Inference Pipeline (Top-k → Temperature → Softmax → Multinomial)
```python
context = "Every effort moves you"
token_ids = torch.tensor(tokenizer.encode(context, allowed_special={'<|endoftext|>'}))
token_ids = token_ids.unsqueeze(0).to(model_device)

max_length, top_k, temperature = 25, 15, 1.4
context_size = GPT_CONFIG_124M["context_length"]

model.eval()
with torch.no_grad():
    for _ in range(max_length):
        logits = model(token_ids[:, -context_size:])           # slice keeps input ≤ context_length
        logits = logits[:, -1, :]                               # last token's logits → (B, 50257)
        top_values, _ = torch.topk(logits, k=top_k)             # descending order by default
        min_val = top_values[:, -1]                             # k-th-largest logit per batch
        logits = torch.where(                                   # NEW: torch.where (was in-place assign)
            condition=logits < min_val,
            input=torch.tensor(float('-inf')),
            other=logits)
        input = logits / temperature                            # temperature AFTER masking
        probs = torch.softmax(input, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token_id], dim=-1)

print(tokenizer.decode(token_ids.squeeze(0).tolist()))
```

Pipeline order: `last logits → top-k mask → /T → softmax → multinomial`.
- **Top-k**: which tokens are candidates (mask non-top-k to -∞)
- **Temperature**: how confidently to pick (T<1 sharper; T>1 flatter; T→0 greedy; T→∞ uniform)
- **Multinomial**: actual random draw — unlike `argmax`, allows lower-prob tokens occasionally

### Inference Runs Twice
- After scratch training: `top_k=15, temperature=1.4` (lines 683-715)
- After loading OpenAI weights: `top_k=25, temperature=1.4` (lines 810-842)

### Loading OpenAI Weights
OpenAI stores Q/K/V concatenated in `c_attn`; bias likewise concatenated in `c_attn['b']`:
```python
w_q, w_k, w_v = np.split(params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
assign(model.trf_blocks[b].att.W_query.weight, w_q.T)   # OpenAI stores (in,out); we use (out,in)
assign(model.trf_blocks[b].att.W_key.weight,   w_k.T)
assign(model.trf_blocks[b].att.W_value.weight, w_v.T)

b_q, b_k, b_v = np.split(params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1)
assign(model.trf_blocks[b].att.W_query.bias, b_q)
assign(model.trf_blocks[b].att.W_key.bias,   b_k)
assign(model.trf_blocks[b].att.W_value.bias, b_v)

assign(model.trf_blocks[b].att.W_out.weight, params['blocks'][b]['attn']['c_proj']['w'].T)
assign(model.trf_blocks[b].att.W_out.bias,   params['blocks'][b]['attn']['c_proj']['b'])

# FFN: c_fc → first linear, c_proj → second linear (with bias)
# LayerNorms: ln_1.g/b → ln1.scale/shift; ln_2.g/b → ln2.scale/shift
# Final: params['g']/['b'] → final_norm.scale/shift; params['wte'] → out_head.weight  (weight tying)
```
Variable names changed from prior version (`q_w, k_w, v_w` → `w_q, w_k, w_v`); bias split is now explicit.

## New / Expanded Pedagogical Sections (vs prior version)

| Section | Lines | Note |
|---|---|---|
| QKV Bias (modern-LLM rationale) | 48-59 | Aligns with [[bias-comparison-gpt2-vs-paper]] |
| Sanity Check on dataset size | 120-135 | Guard against silent empty-dataset training failure |
| `__call__` vs `forward` note | 233-242 | Idiomatic PyTorch reminder |
| Weight Tying detailed walk-through | 174-198 | Mirrors [[weight-tying]] |
| Dropout placement (3× per block) | 346-354 | See updated [[dropout]] page |
| argmax vs multinomial + temperature interplay | 649-656 | Aligns with [[decoding-strategies]] |
| Top-k Sampling (problem + solution) | 658-666+ | Aligns with [[decoding-strategies]] |

## New Concepts

- [[gpt2-from-scratch]]
- [[decoding-strategies]]
- [[weight-tying]]

## Entities

- [[gpt-family]]

## Related

- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[layer-normalization]]
- [[layernorm-scale-shift-sharing]]
- [[layernorm-count-gpt2]]
- [[feed-forward-network]]
- [[kv-caching]]
- [[classification_fine_tuning]]
- [[bias-comparison-gpt2-vs-paper]]
- [[gpt2-parameter-count]]
- [[inference-sliding-window]]
