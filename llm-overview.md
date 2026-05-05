# LLM Internals & NLP — Overview Deck

---

## Slide 1 — Cover

# LLM Internals & NLP

*From Tokens to Generation*

**Dr. Navdeep Singh**
ASSOCIATE PROFESSOR
*Computer Science & Engineering · Punjabi University, Patiala*

---

## Slide 2 — Building Blocks

Every large language model is built from two layers of concepts:

**Foundations**
- Large Language Models — neural nets trained to predict the next word
- Tokenization & BPE — text is broken into subword pieces before the model sees it
- Embeddings — each token is converted into a vector of numbers
- Positional Encodings — tell the model where each token sits in the sequence

**Transformer Core**
- Multi-Head Attention — lets each token look at all other tokens simultaneously
- Layer Normalization — keeps activations stable during training
- Feed-Forward Network (GELU) — adds non-linear transformations after attention
- Residual Connections — let gradients flow directly, preventing vanishing gradients

---

## Slide 3 — GPT-2: Architecture & Training

GPT-2 is a decoder-only transformer trained purely on next-token prediction.

**Architecture**
- Decoder-Only — reads left to right; no encoder; no cross-attention
- Causal Masking — each token can only attend to tokens before it
- Weight Tying — the input embedding matrix is reused as the output projection

**Training**
- Cross-Entropy Loss — the model learns to maximise the probability of the correct next token
- AdamW + LR Warmup — optimizer with weight decay; warmup prevents early divergence
- Scaling Laws — performance improves predictably as we increase parameters, data, and compute

---

## Slide 4 — From Model to Application

Once trained, LLMs are deployed and adapted in three main ways:

**Inference**
- KV Caching — saves computed keys/values so past tokens aren't reprocessed
- Decoding Strategies — greedy, top-k, and nucleus sampling control how text is generated
- Temperature — a single number that controls how random or focused the output is

**Fine-Tuning**
- LoRA — inserts small trainable matrices; the base model stays frozen
- Instruction Fine-Tuning — trains on (prompt, response) pairs to follow user instructions
- Classification Head — the output layer is replaced to map to class labels

**Evaluation**
- Perplexity — measures how surprised the model is by held-out text; lower is better
- BLEU Score — counts n-gram overlap between generated and reference text
- Scaling Laws — emergent abilities (reasoning, coding) appear sharply past certain scale thresholds
