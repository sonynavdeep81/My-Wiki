# Wiki Index

## Navigation
- [Learning Path](learning-path.md) — Canonical reading order across all concept pages (7 stages, 40 concepts, 5 gaps)

## Queries
- [GPT-2 Pretraining Implementation — Study Notes](wiki/queries/gpt2-pretraining-implementation-notes.md) — Q&A session: dataset creation, LayerNorm, MultiHeadAttention, weight tying, shapes, and PyTorch mechanics

## Sources
- [Decoder Architecture (Slide Deck)](wiki/sources/Decoder_archtecture.md) — 51-slide walkthrough of LLM internals from tokenization to the full decoder-only transformer
- [GPT-2 Decoder — Architecture & Pretraining (Python)](wiki/sources/gpt2_decoder.md) — Colab-exported PyTorch script: 2 configs (ctx=256 scratch / ctx=1024 OpenAI), full architecture, sanity check, training, save/load, two inference passes (top-k → /T → softmax → multinomial), OpenAI weight loading w/ bias split
- [Classification Fine-Tuning (Python)](wiki/sources/classification_fine_tuning.md) — GPT-2 spam classifier: SpamDataset, freeze strategy, head replacement (bias=True), training loop, checkpoint save/load, inference
- [Attention Is All You Need (Paper)](wiki/sources/Attention_2023.md) — Vaswani et al. 2017; encoder-decoder Transformer, scaled dot-product attention, sinusoidal PE, ReLU FFN, Post-LN
- [Instruction Fine-Tuning (Notebook)](wiki/sources/instruction_fine_tuning.md) — GPT-2 355M instruction fine-tuning: Alpaca format, custom_collate, EOS stop token in generate(), LLM-as-judge evaluation via Groq
- [Build a Large Language Model (From Scratch)](wiki/sources/Raschka-LLM-2025.md) — Raschka 2025 (Manning); full GPT-2 from-scratch book; parent source for gpt2_decoder, classification, and instruction fine-tuning code

## Queries
- [LLM Workflow — Student Notes](wiki/queries/llm-workflow-student-notes.md) — Complete GPT-2 workflow from tokenization to predicted token; student-friendly with corrections on dropout, FFN, and W_O projection
- [Classification Fine-Tuning Workflow — Spam Detection](wiki/queries/classification-finetuning-workflow.md) — End-to-end walkthrough: balancing, label encoding, tokenization, padding, freeze strategy, and last-token forward pass
- [Padding Strategy — Classification vs Instruction Fine-Tuning](wiki/queries/padding-strategy-classification-vs-instruction.md) — Why classification pads to dataset max while instruction fine-tuning pads per batch; how to swap them
- [Instruction Fine-Tuning — Template Consistency](wiki/queries/instruction-finetuning-template-consistency.md) — Why training and inference templates must be identical; token-pattern reflex explained with training/inference examples and the dog analogy
- [SpamDataset — Truncation and Padding Lines Explained](wiki/queries/spamdataset-truncation-padding-lines.md) — What the two encoded_texts lines do; why truncation comes first; why the crash is in DataLoader not the model; model accepts any length up to 1024

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
- [Fine-Tuning](wiki/concepts/fine-tuning.md) — Instruction vs classification fine-tuning; PEFT family (LoRA, QLoRA, Adapters, etc.); max_tokens consistency rule
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
- [LoRA (Low-Rank Adaptation)](wiki/concepts/lora.md) — Freeze base weights, inject trainable rank-r matrices; <0.1% of params matches full fine-tuning
- [Temperature (Decoding)](wiki/concepts/temperature.md) — Divides logits before softmax; T<1 sharper, T>1 flatter, T→0 greedy, T→∞ uniform
- [Softmax](wiki/concepts/softmax.md) — Converts scores to probabilities; used in attention weights and output head
- [Perplexity](wiki/concepts/perplexity.md) — exp(cross-entropy loss); standard LM eval metric; lower = better
- [Learning Rate Warmup](wiki/concepts/lr-warmup.md) — Gradually ramps LR from 0 to target over first N steps; prevents early divergence
- [Instruction Fine-Tuning](wiki/concepts/instruction-fine-tuning.md) — Alpaca prompt format; dynamic padding via custom_collate; masks padding only (NOT instruction tokens); AdamW 2 epochs; deterministic eval top_k=1
- [Cross-Entropy Loss](wiki/concepts/cross-entropy-loss.md) — -log(p_correct); ignore_index=-100 skips masked positions; do not pre-softmax; exp(CE)=perplexity
- [LLM Evaluation](wiki/concepts/llm-evaluation.md) — Three approaches: benchmarks (MMLU), human preference (LMSYS), automated LLM scoring (AlpacaEval); fair comparison requires same-size models
- [Cosine Decay](wiki/concepts/cosine-decay.md) — LR schedule following half-cosine curve from peak to min_lr after warmup; prevents overshoot of loss minima
- [Gradient Clipping](wiki/concepts/gradient-clipping.md) — Caps gradient L2 norm at max_norm=1.0 to prevent exploding gradients; applied after warmup phase
- [Zero-Shot and Few-Shot Learning](wiki/concepts/zero-shot-few-shot.md) — GPT generalizes to unseen tasks from prompt alone; zero-shot=no examples, few-shot=small demos
- [BERT](wiki/concepts/bert.md) — Encoder-only transformer; bidirectional masked prediction; strong at classification, weak at generation
- [Ablation Study](wiki/concepts/ablation-study.md) — Remove one component at a time, measure the drop; mandatory at top venues; types, examples, what it is NOT

## Entities
- [Attention Is All You Need](wiki/entities/attention-is-all-you-need.md) — Vaswani et al. 2017; encoder-decoder, 8 heads, d_model=512, ReLU FFN, Post-LN, sinusoidal PE
- [Sebastian Raschka](wiki/entities/sebastian-raschka.md) — Author of "Build a Large Language Model"; staff research engineer at Lightning AI
- [Stanford Alpaca](wiki/entities/stanford-alpaca.md) — LLaMA fine-tuned on 52K GPT-3 instruction pairs; popularized Alpaca prompt template; non-commercial
- [tiktoken](wiki/entities/tiktoken.md) — OpenAI's BPE tokenizer library for GPT models
- [LLaMA](wiki/entities/llama.md) — Meta's open-source model family; 405B near GPT-4 on MMLU
- [GPT Family](wiki/entities/gpt-family.md) — OpenAI's decoder-only models GPT-2 through GPT-5.1; architecture details and training data

## Lint
- [Lint — 2026-04-14](wiki/lint/lint-2026-04-14.md) — 1 broken link, 3 under-linked pages, 4 missing concepts, 5 source gaps
- [Lint — 2026-04-14b](wiki/lint/lint-2026-04-14b.md) — Fixed 3 broken wikilinks + 1 orphan; 4 knowledge gaps suggested
- [Lint — 2026-04-14c](wiki/lint/lint-2026-04-14c.md) — Clean: 0 orphans, 0 broken links; 3 gaps flagged (Label Smoothing, AdamW, BLEU)
- [Lint — 2026-04-14d](wiki/lint/lint-2026-04-14d.md) — Fixed 2 orphans (optimizer, label-smoothing); wiki fully clean
- [Lint — 2026-04-16](wiki/lint/lint-2026-04-16.md) — Fixed 6 broken wikilinks; 0 orphan concepts; 5 knowledge gaps flagged (softmax, temperature, LoRA, perplexity, warmup)
- [Lint — 2026-04-30](wiki/lint/lint-2026-04-30.md) — 1 orphan fixed; 2 index errors fixed; Stanford Alpaca entity created; 3 gaps flagged
- [Lint — 2026-04-24](wiki/lint/lint-2026-04-24.md) — 11 orphans fixed; 0 broken links in active pages; 3 gaps flagged
- [Lint — 2026-04-18](wiki/lint/lint-2026-04-18.md) — 0 orphans; 1 broken link fixed (bleu-score backslash); 5 knowledge gaps persist (LoRA priority)
- [Lint — 2026-05-05](wiki/lint/lint-2026-05-05.md) — 2 orphans, 2 raw files to re-check, 2 missing concept pages (Instruction Fine-Tuning, Cross-Entropy Loss)
- [Lint — 2026-05-17](wiki/lint/lint-2026-05-17.md) — structural health check; genuine broken links found (research-p3-sparse-lora, ingest-workflow); capitalized display-name variants confirmed benign

## Queries
- [Input Text to Output Tokens](wiki/queries/input-to-output-workflow.md) — End-to-end workflow with shape trace and ASCII diagram
- [GPT-2 Parameter Count — 124M vs 162M](wiki/queries/gpt2-parameter-count.md) — component breakdown; why model.parameters() double-counts tied weights to show 162M
- [GPT-2 vs Attention Is All You Need — Params](wiki/queries/gpt2-vs-attention-paper-params.md) — Full parameter comparison: your decoder-only GPT-2 vs the original encoder-decoder Transformer
- [Evaluation Metrics for a Decoder-Only LLM](wiki/queries/llm-evaluation-metrics.md) — Loss, perplexity, generation quality, fine-tuning metrics; BLEU/ROUGE not applicable
- [Why .bool() on the Causal Mask](wiki/queries/causal-mask-bool.md) — masked_fill_ requires BoolTensor; also 4× memory saving over int/float
- [Why Use register_buffer?](wiki/queries/register-buffer.md) — Keeps fixed tensors device-aware, checkpoint-included, and optimizer-excluded
- [model.parameters() and p.numel()](wiki/queries/model-parameters-numel.md) — Recursive parameter iteration, element counting, and weight-tying double-count caveat
- [Bias Comparison — GPT-2 vs Attention Is All You Need](wiki/queries/bias-comparison-gpt2-vs-paper.md) — Which layers use bias: scratch vs OpenAI checkpoint vs original Transformer paper
- [Training Loop Primitives](wiki/queries/training-loop-primitives.md) — model.train/eval, zero_grad, backward, optimizer.step, no_grad: what each does and why placed where
- [Why Save the Optimizer State?](wiki/queries/why-save-optimizer-state.md) — AdamW tracks m, v, step count per param; discarding on resume causes loss spikes
- [Why Concept Pages Exist](wiki/queries/why-concept-pages.md) — Speed, consistency, token savings; analogy and examples from our discussion
- [Inference Sliding Window — Handling Context Length During Generation](wiki/queries/inference-sliding-window.md) — pos_emb table fixed at context_length rows; sliding window keeps last 256 tokens; bug in gpt2_decoder.py generate function
- [Context Length Assert — Why max_tokens Must Not Exceed context_length](wiki/queries/context-length-assert.md) — pos_emb table has fixed rows; sequences beyond context_length crash with index out of bounds
- [DataLoader Parameters — shuffle and drop_last](wiki/queries/dataloader-parameters.md) — shuffle prevents order memorization; drop_last discards incomplete final batch
- [requires_grad vs torch.no_grad()](wiki/queries/requires-grad-vs-no-grad.md) — requires_grad freezes specific params permanently; no_grad() suspends all tracking temporarily for inference
- [Classification Fine-Tuning Strategy — What to Freeze and What to Train](wiki/queries/classification-finetuning-strategy.md) — train final head+block+norm; freeze all other 11 blocks; reasons for each
- [SpamDataset — Classification Fine-Tuning Dataset Implementation](wiki/queries/spam-dataset-implementation.md) — PyTorch Dataset pattern: tokenize → truncate → pad → tensor; max_tokens consistency across splits
- [Train vs Val vs Test Split](wiki/queries/train-val-test-split.md) — Model trains on train only; val guides human decisions (indirect leakage); test is final unbiased eval
- [Instruction Fine-Tuning — Data Format (Instruction + Desired Response)](wiki/queries/instruction-finetuning-data-format.md) — Training pairs of (instruction, desired response); loss on response only; contrast with classification fine-tuning
- [Instruction Fine-Tuning — Prompt Format (Alpaca and Others)](wiki/queries/instruction-finetuning-prompt-format.md) — Alpaca template: 3 fields (instruction/input/response); delimiters are plain text strings not special tokens; training format must match inference
- [Instruction Fine-Tuning — Data Preparation Pipeline (5 Steps)](wiki/queries/instruction-finetuning-data-pipeline.md) — format (3 fields) → tokenize → pad+1 → shift → mask; common misunderstandings table
- [Instruction Fine-Tuning — Collate Padding Trick (batch_max_length +1)](wiki/queries/instruction-finetuning-collate-padding-trick.md) — pad to max_len+1; inputs=padded[:-1], targets=padded[1:]; order of operations: max_length first, stop token second
- [Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)](wiki/queries/instruction-finetuning-training-mechanics.md) — dynamic padding, target=input+1, loss masked on padding; instruction tokens not masked in reference impl
- [Instruction Fine-Tuning — Why Only Targets Are Masked (Not Inputs)](wiki/queries/instruction-finetuning-why-only-targets-masked.md) — inputs → model forward pass; targets → loss; -100 = excluded from loss, gradient, and weight update entirely
- [Dropout During Fine-Tuning — Why Set drop_rate=0.0](wiki/queries/dropout-during-finetuning.md) — Dropout noise averages out at pretraining scale (~10⁹ looks) but not at fine-tune scale (~10⁴); turn it off when mostly-frozen + small data
- [How do we evaluate LLMs? (MMLU & comparison strategy)](wiki/queries/llm-evaluation-mmlu.md) — Breadcrumb: 3 evaluation methods; MMLU structure; fair comparison = same-size models
- [[optimizer-zero-grad]] — why zero_grad is needed each training step; gradient accumulation pattern
- [LayerNorm — Are Scale and Shift Shared Across Tokens?](wiki/queries/layernorm-scale-shift-sharing.md) — γ, β shared across all tokens & batch; per-feature `(emb_dim,)`; 1536 params per LayerNorm in GPT-2 124M
- [How Many LayerNorm Layers Does GPT-2 Have?](wiki/queries/layernorm-count-gpt2.md) — `2 × n_layers + 1` formula; GPT-2 124M = 25 LayerNorms = 38,400 params; per-size table
- [What Is an Ablation Study? (Simple Explanation)](wiki/queries/ablation-study-explained.md) — minus-one experiments; ablation table format; difference from tuning/baseline; concrete table you could run on your GPT-2 code
