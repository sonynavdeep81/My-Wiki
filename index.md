# Wiki Index

## Sources
- [Decoder Architecture (Slide Deck)](wiki/sources/Decoder_archtecture.md) — 51-slide walkthrough of LLM internals from tokenization to the full decoder-only transformer
- [GPT-2 From Scratch (Notebook)](wiki/sources/GPT2_Clean.md) — Complete PyTorch implementation: architecture, training, inference, OpenAI weight loading, classification fine-tuning

## Concepts
- [Large Language Models (LLMs)](wiki/concepts/large-language-models.md) — Neural nets trained on next-token prediction; parameters, multimodal, open vs closed
- [Transformer Architecture](wiki/concepts/transformer-architecture.md) — Self-attention neural network; three variants (encoder-decoder, decoder-only, encoder-only)
- [Decoder-Only Architecture](wiki/concepts/decoder-only-architecture.md) — GPT-style transformer; teacher forcing, causal masking, exposure bias
- [Tokenization](wiki/concepts/tokenization.md) — Breaking text into token IDs; word vs character vs subword (BPE)
- [Byte-Pair Encoding (BPE)](wiki/concepts/byte-pair-encoding.md) — Iterative subword merging algorithm; universal LLM tokenization standard
- [Embeddings](wiki/concepts/embeddings.md) — Token + positional lookup tables; 50,257×768 for GPT-2; learned via backprop
- [Multi-Head Attention](wiki/concepts/multi-head-attention.md) — Q/K/V matrices, causal masking, 12 heads of dim 64, contextual vectors
- [Layer Normalization](wiki/concepts/layer-normalization.md) — Stabilizes training; Pre-LN (modern) vs Post-LN (original paper)
- [Feed-Forward Network (FFN)](wiki/concepts/feed-forward-network.md) — Per-token 768→3072 (GELU)→768; adds non-linear features post-attention
- [Scaling Laws](wiki/concepts/scaling-laws.md) — Performance scales with parameters × data × compute; Chinchilla laws
- [Emergent Abilities](wiki/concepts/emergent-abilities.md) — Reasoning/math/coding arise from next-token prediction at scale
- [KV Caching](wiki/concepts/kv-caching.md) — Stores K/V matrices at inference to avoid redundant recomputation
- [Inference Scaling](wiki/concepts/inference-scaling.md) — 2025 paradigm: more compute per query instead of bigger models
- [GPT-2 From-Scratch Patterns](wiki/concepts/gpt2-from-scratch.md) — PyTorch class hierarchy, qkv_bias duality, causal mask buffer, weight tying, OpenAI checkpoint loading
- [Decoding Strategies](wiki/concepts/decoding-strategies.md) — Temperature scaling, top-k sampling, torch.multinomial; quality-creativity trade-off
- [Weight Tying](wiki/concepts/weight-tying.md) — tok_emb and out_head share the same tensor; reduces 38.6M params, improves convergence
- [Fine-Tuning](wiki/concepts/fine-tuning.md) — Instruction vs classification fine-tuning; PEFT, LoRA, QLoRA
- [Positional Embeddings](wiki/concepts/positional-embeddings.md) — Sinusoidal vs learnable (GPT-2) vs RoPE (LLaMA/Mistral); fixes broken wikilink
- [Residual Connections](wiki/concepts/residual-connections.md) — Skip connections preventing vanishing gradients in deep networks
- [Causal Masking](wiki/concepts/causal-masking.md) — Look-ahead mask enforcing autoregressive structure; register_buffer pattern

## Entities
- [Attention Is All You Need](wiki/entities/attention-is-all-you-need.md) — Vaswani et al. 2017 paper introducing the Transformer
- [tiktoken](wiki/entities/tiktoken.md) — OpenAI's BPE tokenizer library for GPT models
- [LLaMA](wiki/entities/llama.md) — Meta's open-source model family; 405B near GPT-4 on MMLU
- [GPT Family](wiki/entities/gpt-family.md) — OpenAI's decoder-only models GPT-2 through GPT-5.1; architecture details and training data

## Queries
- [Input Text to Output Tokens](wiki/queries/input-to-output-workflow.md) — End-to-end workflow with shape trace and ASCII diagram
- [Lint — 2026-04-14](wiki/queries/lint-2026-04-14.md) — 1 broken link, 3 under-linked pages, 4 missing concepts, 5 source gaps
- [Feasible Research Topics](wiki/queries/research-topics-feasible.md) — Layer-wise emergence probing and few-shot LoRA vs full fine-tuning
