---
title: GPT-2 From-Scratch Implementation Patterns
type: concept
tags: [gpt2, pytorch, implementation, architecture, patterns]
sources: 1
updated: 2026-04-13
---

## GPT-2 From-Scratch Implementation Patterns

**Summary**: Concrete PyTorch patterns for building a GPT-2-class decoder-only transformer, covering class hierarchy, the qkv_bias duality, causal masking as a buffer, weight tying, and the OpenAI checkpoint loading dance.

## Class Hierarchy

```
GPT2Model
  ├── tok_emb: nn.Embedding(50257, 768)
  ├── pos_emb: nn.Embedding(context_length, 768)
  ├── dropout: nn.Dropout(0.1)
  ├── trf_blocks: nn.Sequential(12 × TransformerBlock)
  │     ├── ln1: LayerNorm
  │     ├── att: MultiHeadAttention
  │     ├── ln2: LayerNorm
  │     └── ff:  FeedForward (768→3072→768 via GELU)
  ├── final_norm: LayerNorm
  └── out_head: nn.Linear(768, 50257, bias=False)  ← weight-tied to tok_emb
```

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

OpenAI stores Q, K, V concatenated in a single matrix `c_attn` of shape `(768, 2304)`. Must split before assigning:

```python
q_w, k_w, v_w = np.split(params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
assign(model.trf_blocks[b].att.W_query.weight, q_w.T)   # note .T (OpenAI uses column-major)
assign(model.trf_blocks[b].att.W_key.weight,   k_w.T)
assign(model.trf_blocks[b].att.W_value.weight, v_w.T)
```

The final `out_head` is NOT in the checkpoint — it shares `wte`, so it's implicitly loaded.

## Training Setup

- Optimizer: `AdamW(lr=0.0004, weight_decay=0.1)`
- Loss: `nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten())`
- Dataset: stride-based sliding window (`GPTDataset`), 90/10 train/val split
- Checkpointing: save both `model_state_dict` AND `optimizer_state_dict` to resume training

## Approximate Parameter Count (124M config)

- Token embeddings: 50,257 × 768 ≈ 38.6M
- Positional embeddings: 256 × 768 ≈ 0.2M
- 12 × TransformerBlock:
  - Attention (Q, K, V, W_out): 4 × 768² ≈ 2.36M each
  - FFN (two linear layers): 768×3072 + 3072×768 ≈ 4.72M each
  - LayerNorms: negligible
- Total: ~162M (slightly more than 124M due to context_length=256 vs 1024)

## Related

- [[decoder-only-architecture]]
- [[multi-head-attention]]
- [[weight-tying]]
- [[decoding-strategies]]
- [[fine-tuning]]
- [[layer-normalization]]
