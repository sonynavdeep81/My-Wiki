# Wiki Index

## Sources
- [Decoder Architecture (Slide Deck)](wiki/sources/Decoder_archtecture.md) — 51-slide walkthrough of LLM internals from tokenization to the full decoder-only transformer
- [GPT-2 From Scratch (Notebook)](wiki/sources/GPT2_Clean.md) — Complete PyTorch implementation: architecture, training, inference, OpenAI weight loading, classification fine-tuning
- [Attention Is All You Need (Paper)](wiki/sources/Attention_2023.md) — Vaswani et al. 2017; encoder-decoder Transformer, scaled dot-product attention, sinusoidal PE, ReLU FFN, Post-LN

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
- [Cross-Attention](wiki/concepts/cross-attention.md) — Decoder queries attend to encoder K/V; absent in decoder-only models like GPT-2
- [PyTorch nn Building Blocks](wiki/concepts/pytorch-nn-building-blocks.md) — nn.Module base class; nn.Parameter vs nn.Linear vs nn.Embedding vs nn.Sequential compared
- [GELU Activation](wiki/concepts/gelu.md) — Smooth probabilistic activation used in GPT-2 FFN; softer alternative to ReLU
- [Dropout](wiki/concepts/dropout.md) — Regularization via random zeroing of activations; disabled at inference via model.eval()
- [Label Smoothing](wiki/concepts/label-smoothing.md) — Soft training targets to prevent overconfidence; used in paper (ε=0.1), not in GPT-2 notebook
- [Adam and AdamW Optimizers](wiki/concepts/optimizer.md) — Adaptive gradient optimizers; AdamW fixes weight decay coupling; your GPT-2 uses AdamW, paper uses Adam
- [BLEU Score](wiki/concepts/bleu-score.md) — N-gram overlap metric for machine translation; used in paper results, not applicable to GPT-2

## Entities
- [Attention Is All You Need](wiki/entities/attention-is-all-you-need.md) — Vaswani et al. 2017; encoder-decoder, 8 heads, d_model=512, ReLU FFN, Post-LN, sinusoidal PE
- [tiktoken](wiki/entities/tiktoken.md) — OpenAI's BPE tokenizer library for GPT models
- [LLaMA](wiki/entities/llama.md) — Meta's open-source model family; 405B near GPT-4 on MMLU
- [GPT Family](wiki/entities/gpt-family.md) — OpenAI's decoder-only models GPT-2 through GPT-5.1; architecture details and training data

## Lint
- [Lint — 2026-04-14](wiki/lint/lint-2026-04-14.md) — 1 broken link, 3 under-linked pages, 4 missing concepts, 5 source gaps
- [Lint — 2026-04-14b](wiki/lint/lint-2026-04-14b.md) — Fixed 3 broken wikilinks + 1 orphan; 4 knowledge gaps suggested
- [Lint — 2026-04-14c](wiki/lint/lint-2026-04-14c.md) — Clean: 0 orphans, 0 broken links; 3 gaps flagged (Label Smoothing, AdamW, BLEU)
- [Lint — 2026-04-14d](wiki/lint/lint-2026-04-14d.md) — Fixed 2 orphans (optimizer, label-smoothing); wiki fully clean
- [Lint — 2026-04-16](wiki/lint/lint-2026-04-16.md) — Fixed 6 broken wikilinks; 0 orphan concepts; 5 knowledge gaps flagged (softmax, temperature, LoRA, perplexity, warmup)

## Queries
- [Input Text to Output Tokens](wiki/queries/input-to-output-workflow.md) — End-to-end workflow with shape trace and ASCII diagram
- [Feasible Research Topics](wiki/queries/research-topics-feasible.md) — Layer-wise emergence probing and few-shot LoRA vs full fine-tuning
- [GPT-2 vs Attention Is All You Need — Params](wiki/queries/gpt2-vs-attention-paper-params.md) — Full parameter comparison: your decoder-only GPT-2 vs the original encoder-decoder Transformer
- [Evaluation Metrics for a Decoder-Only LLM](wiki/queries/llm-evaluation-metrics.md) — Loss, perplexity, generation quality, fine-tuning metrics; BLEU/ROUGE not applicable
- [Why .bool() on the Causal Mask](wiki/queries/causal-mask-bool.md) — masked_fill_ requires BoolTensor; also 4× memory saving over int/float
- [Why Use register_buffer?](wiki/queries/register-buffer.md) — Keeps fixed tensors device-aware, checkpoint-included, and optimizer-excluded
- [model.parameters() and p.numel()](wiki/queries/model-parameters-numel.md) — Recursive parameter iteration, element counting, and weight-tying double-count caveat
- [Bias Comparison — GPT-2 vs Attention Is All You Need](wiki/queries/bias-comparison-gpt2-vs-paper.md) — Which layers use bias: scratch vs OpenAI checkpoint vs original Transformer paper
- [Training Loop Primitives](wiki/queries/training-loop-primitives.md) — model.train/eval, zero_grad, backward, optimizer.step, no_grad: what each does and why placed where
- [Why Save the Optimizer State?](wiki/queries/why-save-optimizer-state.md) — AdamW tracks m, v, step count per param; discarding on resume causes loss spikes
