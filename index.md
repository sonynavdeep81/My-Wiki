# Wiki Index

## Navigation
- [Learning Path](learning-path.md) — Canonical reading order across all concept pages (7 stages, 34 concepts, 10 gaps)

## Sources
- [Decoder Architecture (Slide Deck)](wiki/sources/Decoder_archtecture.md) — 51-slide walkthrough of LLM internals from tokenization to the full decoder-only transformer
- [GPT-2 Decoder — Architecture & Pretraining (Python)](wiki/sources/gpt2_decoder.md) — PyTorch script: data batching, model architecture, pretraining, inference (temperature/top-k/multinomial), OpenAI weight loading
- [Classification Fine-Tuning (Python)](wiki/sources/classification_fine_tuning.md) — GPT-2 spam classifier: SpamDataset, freeze strategy, head replacement (bias=True), accuracy training loop
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
- [Instruction Fine-Tuning](wiki/concepts/instruction-fine-tuning.md) — (instruction, response) training pairs; loss on response tokens only; dynamic padding; 5-step data pipeline
- [Cross-Entropy Loss](wiki/concepts/cross-entropy-loss.md) — -log(p_correct); ignore_index=-100 skips masked positions; do not pre-softmax; exp(CE)=perplexity

## Entities
- [Attention Is All You Need](wiki/entities/attention-is-all-you-need.md) — Vaswani et al. 2017; encoder-decoder, 8 heads, d_model=512, ReLU FFN, Post-LN, sinusoidal PE
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

## Queries
- [Input Text to Output Tokens](wiki/queries/input-to-output-workflow.md) — End-to-end workflow with shape trace and ASCII diagram
- [Research P3 — LoRA Placement Transferability](wiki/queries/research-p3-sparse-lora.md) — **Active paper.** Cross-task layer-placement transferability for LoRA in small LMs; 3 contributions, 8-week plan, target IEEE Access
- [P3 Study Guide (student-friendly)](wiki/queries/research-p3-study-guide.md) — Plain-English companion to P3; explains gap, work, and what to study in what order; start here before the technical file
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
- [Student Paper S1 — LoRA on Hinglish Code-Mixed Tasks](wiki/queries/research-student-hinglish-lora.md) — **Do first.** Placement + rank study on COMI-LINGUA with XLM-R/MuRIL; 4–5 weeks, UGC; India-niche novelty
- [Student Paper S2 — Layer-Importance Method Comparison](wiki/queries/research-student-layer-importance-comparison.md) — **Do after S1.** Head-to-head of 4 layer-scoring methods on small LMs; 5 weeks; directly supports P3's LOLO
- [Train vs Val vs Test Split](wiki/queries/train-val-test-split.md) — Model trains on train only; val guides human decisions (indirect leakage); test is final unbiased eval
- [Instruction Fine-Tuning — Collate Padding Trick (batch_max_length +1)](wiki/queries/instruction-finetuning-collate-padding-trick.md) — pad to max_len+1; inputs=padded[:-1], targets=padded[1:]; extra token becomes last target position
- [Instruction Fine-Tuning — Data Preparation Pipeline (5 Steps)](wiki/queries/instruction-finetuning-data-pipeline.md) — format → tokenize → pad → shift target left+append 50256 → replace padding with -100
- [Instruction Fine-Tuning — Training Mechanics (Padding, Shift, Loss Masking)](wiki/queries/instruction-finetuning-training-mechanics.md) — dynamic padding, target=input+1, loss masked on instruction; only response tokens graded
- [Instruction Fine-Tuning — Data Format (Instruction + Desired Response)](wiki/queries/instruction-finetuning-data-format.md) — Training pairs of (instruction, desired response); loss on response only; contrast with classification fine-tuning
- [Instruction Fine-Tuning — Prompt Format (Alpaca and Others)](wiki/queries/instruction-finetuning-prompt-format.md) — No universal standard; Alpaca template popularized but training/inference format must match
- [Dropout During Fine-Tuning — Why Set drop_rate=0.0](wiki/queries/dropout-during-finetuning.md) — Dropout noise averages out at pretraining scale (~10⁹ looks) but not at fine-tune scale (~10⁴); turn it off when mostly-frozen + small data
