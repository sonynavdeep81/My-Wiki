---
title: Learning Path
type: reference
updated: 2026-05-05
---

# Learning Path — LLM Internals & NLP

Canonical reading order. Each stage builds on the previous.

---

## Stage 1: Foundations

- [[Large Language Models]]
- [[Zero-Shot and Few-Shot Learning]]
- [[Tokenization]]
- [[Byte-Pair Encoding]]
- [[Embeddings]]
- [[Positional Embeddings]]

## Stage 2: Transformer Internals

- [[Transformer Architecture]]
- [[BERT]]
- [[Softmax]]
- [[Multi-Head Attention]]
- [[Cross-Attention]]
- [[Causal Masking]]
- [[Feed-Forward Network]]
- [[GELU Activation]]
- [[Layer Normalization]]
- [[Residual Connections]]
- [[Dropout]]

## Stage 3: GPT-2 Architecture

- [[Decoder-Only Architecture]]
- [[GPT-2 From-Scratch Implementation Patterns]]
- [[Weight Tying]]
- [[PyTorch NN Building Blocks]]

## Stage 4: Training Mechanics

- [[Cross-Entropy Loss]]
- [[Optimizer]]
- [[LR Warmup]]
- [[Cosine Decay]]
- [[Gradient Clipping]]
- [[Label Smoothing]]

## Stage 5: Inference & Decoding

- [[KV Caching]]
- [[Decoding Strategies]]
- [[Temperature]]
- [[Inference Scaling]]

## Stage 6: Fine-Tuning & Adaptation

- [[Fine-Tuning]]
- [[Instruction Fine-Tuning]]
- [[LoRA]]

## Stage 7: Evaluation & Scaling

- [[Perplexity]]
- [[BLEU Score]]
- [[Ablation Study]]
- [[Scaling Laws]]
- [[Emergent Abilities]]

---

## Query Reading Order

Q&A files in the order they should be read — each group assumes the previous group is done. New queries are inserted here at the correct position when filed.

### Group A — Meta & Basic ML (no prerequisites)
- [[why-concept-pages]] — how this wiki works and why it is structured this way
- [[train-val-test-split]] — train / val / test splits and why all three are needed

### Group B — PyTorch Foundations
- [[model-parameters-numel]] — what model.parameters() yields and how to count params
- [[register-buffer]] — why register_buffer exists vs nn.Parameter vs plain attribute
- [[dataloader-parameters]] — shuffle, drop_last, and when to use each

### Group C — GPT-2 Architecture & Data Pipeline
- [[llm-workflow-student-notes]] — complete end-to-end LLM workflow with examples (start here)
- [[input-to-output-workflow]] — concise shape-traced workflow from text to token
- [[gpt2-pretraining-implementation-notes]] — full implementation Q&A covering all components
- [[context-length-assert]] — why sequences must not exceed context_length
- [[causal-mask-bool]] — why .bool() on the causal mask and how masked_fill works
- [[bias-comparison-gpt2-vs-paper]] — which layers use bias in GPT-2 vs the original paper
- [[gpt2-vs-attention-paper-params]] — architecture comparison: GPT-2 vs Attention Is All You Need
- [[gpt2-parameter-count]] — why the model shows 162M instead of 124M, and how to fix it
- [[layernorm-scale-shift-sharing]] — γ and β are shared across all tokens; per-feature, not per-token; 1536 params total
- [[layernorm-count-gpt2]] — GPT-2 124M has 25 LayerNorms (2×n_layers + 1); each block keeps its own (γ, β); ~38K params total

### Group D — Training Mechanics
- [[optimizer-zero-grad]] — why zero_grad is needed, what happens without it, gradient accumulation
- [[training-loop-primitives]] — zero_grad, backward, step, train/eval, no_grad explained
- [[requires-grad-vs-no-grad]] — freezing parameters vs suspending gradient tracking
- [[why-save-optimizer-state]] — why optimizer state must be checkpointed alongside weights
- [[spam-dataset-implementation]] — SpamDataset: tokenization, truncation, padding
- [[inference-sliding-window]] — how to handle sequences longer than context_length at inference

### Group E — Evaluation
- [[llm-evaluation-metrics]] — loss, perplexity, train/val gap during pretraining
- [[llm-evaluation-mmlu]] — MMLU, human eval, LLM-as-judge, and fair comparison rules
- [[ablation-study-explained]] — what an ablation study is; why every paper needs one; concrete table you could run on your GPT-2 build

### Group F — Fine-Tuning
- [[classification-finetuning-workflow]] — end-to-end spam detection pipeline: balance → tokenize → pad → freeze → last-token forward pass
- [[padding-strategy-classification-vs-instruction]] — why classification uses static dataset-wide padding while instruction FT uses per-batch dynamic padding
- [[spamdataset-truncation-padding-lines]] — truncation then padding; why order matters; crash is in DataLoader not model; model accepts any length up to 1024
- [[classification-finetuning-strategy]] — what to freeze and what to train for classification
- [[dropout-during-finetuning]] — why drop_rate=0.0 during partial fine-tuning
- [[instruction-finetuning-data-format]] — instruction + response pair format
- [[instruction-finetuning-prompt-format]] — Alpaca and other prompt templates
- [[instruction-finetuning-data-pipeline]] — 5-step data preparation pipeline
- [[instruction-finetuning-collate-padding-trick]] — batch_max_length +1 padding trick
- [[instruction-finetuning-training-mechanics]] — dynamic padding, loss masking, instruction tokens

---

## Gaps in This Path

Topics that belong in the sequence but have no concept page yet:

- **Stage 2**: Sparse attention, relative positional encodings (RoPE, ALiBi)
- **Stage 4**: Mixed precision training (fp16/bf16)
- **Stage 5**: Beam search internals, speculative decoding
- **Stage 6**: RLHF, DPO, QLoRA, prefix tuning
- **Stage 7**: ROUGE score, MT-Bench, MMLU evaluation
