---
title: GPT-2 Pretraining Implementation — Study Notes
type: query
tags: [gpt2, pretraining, pytorch, implementation, layernorm, attention, weight-tying]
sources: 1
updated: 2026-05-14
---

## GPT-2 Pretraining Implementation — Study Notes

**Summary**: A complete walkthrough of the GPT-2 pretraining code — from dataset creation to model architecture — explained in simple language with examples.

---

## 1. Setting the Random Seed

```python
torch.manual_seed(123)
```

When PyTorch creates random numbers (for weight initialization, dropout, etc.), it normally produces different values every time you run the script. This makes experiments hard to compare.

`torch.manual_seed(123)` fixes the starting point of PyTorch's random number generator. From that point, the same sequence of random numbers is produced every single time — in the same order.

**Important:** The seed does not reset between calls. It starts a deterministic sequence:

```python
torch.manual_seed(123)
a = torch.rand(2)   # always [0.2961, 0.5166]
b = torch.rand(2)   # always [0.2517, 0.6886]  ← next in the sequence
```

`a` and `b` get different values, but those exact values are identical on every run.

**Why it matters for GPT-2:** Weight initialization uses random numbers. With the seed fixed, your model always starts with identical weights — so two training runs are directly comparable.

> For full reproducibility also set `torch.cuda.manual_seed(123)` for GPU runs.

---

## 2. QKV Biases

```python
GPT_CONFIG_124M = {
    ...
    'qkv_bias': True
}
```

Inside MultiHeadAttention, three linear layers compute Q, K, and V:

```python
self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
self.W_key   = nn.Linear(d_model, d_model, bias=qkv_bias)
self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
```

When `bias=True`, each projection adds a learned bias term:
```
Q = x @ W_q + b_q
K = x @ W_k + b_k
V = x @ W_v + b_v
```

**Why True here?** OpenAI's original GPT-2 weights include these bias terms. If you set `bias=False` and try to load OpenAI's pretrained weights, the shapes don't match and you get a crash.

**Why modern LLMs use False:** Research showed these biases add parameters without meaningfully improving performance. LLaMA, Mistral, and Falcon all drop QKV biases.

---

## 3. UTF-8 Encoding and File Storage

```python
text_data = response.read().decode('utf-8')   # bytes → string
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(text_data)
```

**Key idea:** Files on disk always store bytes — never raw text. Text is a human concept. The computer only understands bytes.

The round-trip works like this:
```
Download:  bytes (UTF-8) → .decode('utf-8') → Python string in memory
Write:     Python string → open(..., encoding='utf-8') → bytes on disk
```

You do not need to manually encode the string before writing. The `encoding='utf-8'` in `open()` tells Python which format to use when converting the string back to bytes — Python handles it automatically.

**Why specify encoding explicitly?** If you omit it, Python uses the system default — which could be `cp1252` on Windows. That may corrupt characters like curly quotes or accented letters. Always specify `utf-8` to be safe.

---

## 4. Sliding Window Dataset

```python
for i in range(0, len(token_ids) - context_length, stride):
    input_tokens  = token_ids[i : i + context_length]
    target_tokens = token_ids[i+1 : i + context_length + 1]
```

This creates training samples for **next-token prediction** — for every input sequence, the target is the same sequence shifted one position to the right.

**Example with context_length=4, stride=1:**
```
token_ids = [A, B, C, D, E, F, G, H]

i=0:  input  = [A, B, C, D]
      target = [B, C, D, E]

i=1:  input  = [B, C, D, E]
      target = [C, D, E, F]
```

Notice that `E` appears as the last target in `i=0` and as the last input in `i=1` — windows overlap.

**Example with stride=4 (no overlap):**
```
i=0:  input  = [A, B, C, D]    target = [B, C, D, E]
i=4:  input  = [E, F, G, H]    target = [F, G, H, I]
```

**Why `- context_length` in the range?** The target always needs one token ahead of the input window. This guard ensures the last window always has a valid target.

**How many iterations?** With 10 tokens, context=4, stride=4:
```
range(0, 10-4, 4) = range(0, 6, 4) → [0, 4]  → 2 iterations only
```
The last 2 tokens are unused — not enough to form a complete window.

**stride=1 vs stride=context_length:**

| | stride=1 | stride=context_length |
|---|---|---|
| Overlap | Maximum | None |
| Training samples | Many | Few |
| Redundancy | High | None |
| Best for | Small datasets | Large datasets |

---

## 5. Dataset Size Check

```python
if len(train_tokens) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for training loader...")
```

This is a **minimum viability check** — it verifies the entire dataset has enough tokens to form at least one training window.

If total tokens < context_length, the sliding window loop produces zero samples, the DataLoader has nothing to load, and training fails silently.

This is separate from the position embedding limit — individual sequences are always exactly `context_length` tokens long (the sliding window guarantees it). This check is about the total dataset size, not individual samples.

---

## 6. DataLoader Setup

```python
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,  drop_last=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, drop_last=False, num_workers=0)
```

**Why these settings:**

| Setting | train | val | Reason |
|---|---|---|---|
| shuffle | True | False | Randomize training order each epoch; validation must be reproducible |
| drop_last | True | False | Keep uniform batch sizes for training; use every validation sample |
| num_workers | 0 | 0 | Safe for Colab/Windows; increase for large-scale training |

**`__len__` and `__getitem__`:**
- `__len__` returns the total number of **samples** (not batches). DataLoader uses this to calculate how many batches to make.
- `__getitem__` fetches one sample at a time by index. DataLoader calls this repeatedly and groups results into batches.

---

## 7. tiktoken and allowed_special

```python
token_ids = tokenizer.encode(text_data, allowed_special={'<|endoftext|>'})
```

`<|endoftext|>` is a special token (ID 50256) that marks boundaries between documents.

Without `allowed_special`, tiktoken raises a `DisallowedSpecialTokenError` — a deliberate safety measure to prevent you from accidentally splitting a special token into regular sub-tokens.

| Call | Behavior |
|---|---|
| `encode(text)` | Error — special tokens disallowed by default |
| `encode(text, allowed_special=set())` | Encodes as multiple regular tokens — loses boundary meaning |
| `encode(text, allowed_special={'<|endoftext|>'})` | Single token ID 50256 ✓ |

If `<|endoftext|>` got split into individual pieces like `<`, `\|`, `end`, `of`, `text`, `\|`, `>`, the model would never learn to recognize it as a meaningful document boundary.

---

## 8. GPT2Model — Forward Pass Shapes

```python
def forward(self, x):
    batch_size, num_tokens = x.shape          # (2, 256)
    tok_emb = self.tok_emb(x)                 # (2, 256, 768)
    pos_emb = self.pos_emb(torch.arange(...)) # (256, 768)
    x = tok_emb + pos_emb                     # (2, 256, 768)  ← broadcast
    x = self.dropout(x)                       # (2, 256, 768)
    x = self.trf_blocks(x)                    # (2, 256, 768)
    x = self.final_norm(x)                    # (2, 256, 768)
    logits = self.out_head(x)                 # (2, 256, 50257)
```

**Why `device=x.device` on `torch.arange`?**

`torch.arange` creates tensors on CPU by default. If your input `x` is on GPU, adding `tok_emb + pos_emb` would fail — PyTorch requires both tensors on the same device. `device=x.device` makes `pos_emb` follow wherever `x` lives — CPU or GPU — automatically.

**The output shape `(2, 256, 50257)`:**
The model produces vocabulary scores for **every token position**, not just the last one. For batch_size=2, context_length=256, that's 512 next-token predictions per forward pass.
- During training: all 512 are used to compute loss
- During inference: only the last position's scores are used

**Why call `model(x)` not `model.forward(x)`?**
`nn.Module.__call__` internally calls `forward` but also handles hooks, gradient tracking, and other PyTorch internals. Always use `model(x)`.

---

## 9. LayerNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, cfg, eps=1e-5):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(cfg['emb_dim']))
        self.shift = nn.Parameter(torch.zeros(cfg['emb_dim']))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std  = torch.std(x,  dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift
```

**What it does:** Normalizes each token's embedding so values have mean=0 and std=1, then applies learnable scale and shift so the model can adjust.

**Why `dim=-1`?** Normalizes across the embedding dimension (768) independently for each token. Each of the 256 tokens gets its own mean and std — they are not mixed across tokens or batches.

**Why `keepdim=True`?**
Without it, mean and std lose their last dimension:
```
x shape:    (2, 256, 768)
mean:       (2, 256)       ← can't subtract from (2, 256, 768)
```
With `keepdim=True`:
```
mean:       (2, 256, 1)    ← broadcasts correctly across 768 dims ✓
```

**Why scale starts at 1, shift at 0?**
At the start of training, `1 × normalized + 0 = normalized` — pure normalization, no change. As training progresses, scale and shift adjust to whatever values help the model learn.

**Why eps (1e-5)?**
If std=0 (all embedding values identical — rare but possible), dividing by zero gives infinity or NaN. NaN spreads to every subsequent calculation, destroying training completely. Adding a tiny `eps` ensures the denominator is never zero.

**Why `super().__init__()`?**
Activates PyTorch's `nn.Module` machinery. Without it, `nn.Parameter` assignments are not tracked — they won't appear in `model.parameters()`, the optimizer never sees them, and scale/shift stay frozen forever. LayerNorm would learn nothing.

**Why `nn.Parameter`?**
Wraps a tensor and marks it as trainable. The optimizer updates anything registered as `nn.Parameter`. A plain `torch.tensor` would be ignored by the optimizer entirely.

---

## 10. FeedForward Network

```python
self.layers = nn.Sequential(
    nn.Linear(emb_dim, 4 * emb_dim),   # 768 → 3072
    GELU(),
    nn.Linear(4 * emb_dim, emb_dim)    # 3072 → 768
)
```

**Shape flow:**
```
Input   (2, 256, 768)
Linear1 (2, 256, 3072)   ← expand by 4×
GELU    (2, 256, 3072)   ← applied element-wise to all 3072 values per token
Linear2 (2, 256, 768)    ← compress back down
```

**Why expand then compress?** The wider middle layer gives the model more capacity to learn complex transformations. After the transformation, it compresses back to the original size so the residual connection still works.

**Indexing:** `ff.layers[0]` = first Linear, `ff.layers[1]` = GELU, `ff.layers[2]` = second Linear.

---

## 11. Dropout in TransformerBlock

Dropout appears **3 times** per transformer block:

```python
# 1. Inside MultiHeadAttention — on attention weights
att_weights = self.dropout(att_weights)

# 2. After MHA output — before residual addition
x = self.att(x)
x = self.dropout(x)
x = x + shortcut

# 3. After FeedForward output — before residual addition
x = self.ff(x)
x = self.dropout(x)
x = x + shortcut
```

| Location | What it drops | Effect |
|---|---|---|
| Attention weights | Token-to-token connections | Forces model not to over-rely on specific attention patterns |
| MHA output | Output values | Regularizes the attention block |
| FF output | Output values | Regularizes the feedforward block |

All three use `drop_rate=0.1` — 10% of values are randomly zeroed during training. At inference time, all dropout is automatically disabled.

---

## 12. MultiHeadAttention — Shapes

```python
Q = self.W_query(x).reshape(batch, tokens, n_heads, head_dim)  # (2,256,12,64)
Q = Q.transpose(1, 2)                                           # (2,12,256,64)

att_scores = Q @ K.transpose(-1, -2)                            # (2,12,256,256)
att_scores = att_scores / (self.head_dim ** 0.5)
att_scores = att_scores.masked_fill(causal_mask, -torch.inf)
att_weights = torch.softmax(att_scores, dim=-1)                 # (2,12,256,256)
context_vec = att_weights @ V                                   # (2,12,256,64)
context_vec = context_vec.transpose(1,2).reshape(batch, tokens, d_model)  # (2,256,768)
```

**What is `Q @ K.transpose(-1, -2)`?**
This computes the dot product between every query and every key — measuring how much each token should attend to every other token. The result `(2, 12, 256, 256)` is a 256×256 similarity matrix per head per batch item.

```
        key0  key1  key2  ...
query0 [ 2.1  0.3  -1.2  ... ]   ← how much token 0 attends to each other token
query1 [ 0.5  3.2   0.1  ... ]
...
```

**Why not `torch.dot()`?** `torch.dot` only works on 1D vectors. `@` works on any dimensionality and broadcasts over all heads and batches simultaneously — equivalent to nested loops but much faster.

**Why `head_dim ** 0.5` not `torch.sqrt(torch.tensor(head_dim))`?** `torch.sqrt` requires a tensor input — passing a plain integer causes an error. `** 0.5` works directly on a Python integer and is simpler.

**Causal mask slicing:** The mask is pre-built for the full `context_length` but sliced to the actual sequence length: `causal_mask[:num_tokens, :num_tokens]`. This handles sequences shorter than context_length correctly.

---

## 13. Weight Tying

**The problem:** Without weight tying, `tok_emb` and `out_head` are two separate 50257×768 matrices — ~38.6M parameters each, counted twice → ~162M total instead of 124M.

**The fix:**
```python
self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
self.out_head.weight = self.tok_emb.weight   # same tensor, two names
```

Both names now point to the **same memory location** — not a copy. The parameter is counted once → ~124M total.

**How updates work:** During backpropagation, gradients flow through both `out_head` (last layer) and `tok_emb` (first layer). Both gradients accumulate into the single shared tensor. The optimizer updates it once — both layers reflect the change instantly.

**Why it makes sense:**
- `tok_emb` learns: token ID → 768-dim vector (lookup)
- `out_head` learns: 768-dim vector → 50257 vocabulary scores (matrix multiply)

These are inverse operations. Sharing weights forces the model to learn a single representation that works well for both encoding tokens and predicting tokens — a meaningful constraint, not a limitation.

**Without weight tying:** Two separate matrices with different values, both evolving independently. More parameters but no significant accuracy gain in practice. GPT-2, GPT-3, LLaMA, and Mistral all use weight tying.

**Important:** `out_head` must still be created as `nn.Linear` — weight tying does not replace it. The linear layer is needed to perform the matrix multiplication in `forward`. Weight tying only makes both layers share the same numbers.

---

## Related

- [[MultiHeadAttention]]
- [[LayerNorm]]
- [[GPT-2 Architecture]]
- [[FeedForward Network]]
- [[Positional Embedding]]
- [[Tokenization]]
