---
title: GPT-2 From-Scratch Implementation Patterns
type: concept
tags: [gpt2, pytorch, implementation, architecture, patterns]
sources: 2
updated: 2026-05-14
verified_against: gpt2_decoder, 2026-05-14
confidence: high
---

## GPT-2 From-Scratch Implementation Patterns

**Summary**: Concrete PyTorch patterns for building a GPT-2-class decoder-only transformer, covering class hierarchy, the qkv_bias duality, causal masking as a buffer, weight tying, and the OpenAI checkpoint loading dance.

## Class Hierarchy

```
GPT2Model
  ├── tok_emb: nn.Embedding(50257, 768)
  ├── pos_emb: nn.Embedding(context_length, 768)   # context_length=256 (scratch) or 1024 (OpenAI)
  ├── dropout: nn.Dropout(0.1)                     # post-embedding
  ├── trf_blocks: nn.Sequential(12 × TransformerBlock)
  │     ├── ln1: LayerNorm
  │     ├── att: MultiHeadAttention                # 3 dropout sites total per block
  │     ├── dropout (post-att, pre-residual)
  │     ├── ln2: LayerNorm
  │     ├── ff:  FeedForward (768→3072→768 via GELU)
  │     └── dropout (post-ff, pre-residual)
  ├── final_norm: LayerNorm
  └── out_head: nn.Linear(768, 50257, bias=False)  ← weight-tied to tok_emb
```

The script ships TWO `GPT_CONFIG_124M` blocks: one with `context_length=256` for scratch training on the demo corpus, one with `context_length=1024` for loading OpenAI checkpoints (full GPT-2 small context window). Both use `qkv_bias=True`.

## qkv_bias Duality

| Scenario | Setting | Reason |
|---|---|---|
| Training from scratch | `qkv_bias=False` | [[layer-normalization]]'s β already handles offset; explicit bias is redundant |
| Loading OpenAI weights | `qkv_bias=True` | OpenAI's checkpoint contains bias tensors; model must have slots to receive them |

This is a common gotcha — setting the wrong value causes a shape mismatch crash when loading weights.

## Causal Mask as register_buffer

```python
self.register_buffer('causal_mask',
    torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
```

- Non-trainable: not updated by optimizer
- Device-aware: automatically moves to GPU with `model.to(device)`
- Upper-triangle = True (future positions) → masked to −∞ in forward pass

## Weight Tying

The output head (`out_head`) shares weights with the token embedding table (`tok_emb`):

```python
# In load_weights_into_gpt2():
assign(model.tok_emb.weight, params['wte'])
assign(model.out_head.weight, params['wte'])  # same tensor
```

See [[weight-tying]] for why this works and is beneficial.

## Loading OpenAI Checkpoints

OpenAI stores Q, K, V concatenated in a single matrix `c_attn['w']` of shape `(768, 2304)` and biases in `c_attn['b']` of shape `(2304,)`. Both must be split before assigning:

```python
# Weights
w_q, w_k, w_v = np.split(params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
assign(model.trf_blocks[b].att.W_query.weight, w_q.T)   # OpenAI: (in,out); ours: (out,in)
assign(model.trf_blocks[b].att.W_key.weight,   w_k.T)
assign(model.trf_blocks[b].att.W_value.weight, w_v.T)

# Biases (because qkv_bias=True)
b_q, b_k, b_v = np.split(params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1)
assign(model.trf_blocks[b].att.W_query.bias, b_q)
assign(model.trf_blocks[b].att.W_key.bias,   b_k)
assign(model.trf_blocks[b].att.W_value.bias, b_v)
```

The final `out_head` is NOT in the checkpoint — it shares `wte`, so it's implicitly loaded:
```python
assign(model.out_head.weight, params['wte'])   # weight tying
```

## Training Setup

- Optimizer: `AdamW(lr=0.0004, weight_decay=0.1)`
- Loss: `nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten())`
- Dataset: stride-based sliding window (`GPTDataset`), 90/10 train/val split
- Checkpointing: save both `model_state_dict` AND `optimizer_state_dict` to resume training

## Approximate Parameter Count (124M config)

- Token embeddings: 50,257 × 768 ≈ 38.6M
- Positional embeddings: `context_length × 768` (≈ 0.2M at ctx=256, ≈ 0.79M at ctx=1024)
- 12 × TransformerBlock:
  - Attention (Q, K, V, W_out): 4 × 768² ≈ 2.36M each
  - FFN (two linear layers): 768×3072 + 3072×768 ≈ 4.72M each
  - LayerNorms: ~1.5K each (negligible)
- **True total with weight tying: ~124M** (out_head shares the 38.6M tok_emb tensor)
- **`sum(p.numel())` reports ~162M** because `model.parameters()` double-counts the tied tensor

The 162M-vs-124M gap is **not** about context_length — it is the weight-tying double-count. See [[gpt2-parameter-count]] and [[weight-tying]].

## Related

- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[weight-tying]]
- [[decoding-strategies]]
- [[fine-tuning]]
- [[layer-normalization]]
- [[pytorch-nn-building-blocks|PyTorch nn Building Blocks]]
- [[optimizer|Adam and AdamW Optimizers]]
- [[lr-warmup]]
- [[llm-evaluation-metrics]]
- [[bias-comparison-gpt2-vs-paper]]
- [[gpt2-vs-attention-paper-params]]
