---
title: Complete LLM Workflow — From Input Sentence to Output Token
type: query
tags: [transformer, gpt2, workflow, student-notes]
updated: 2026-05-14
---

# Complete LLM Workflow — From Input Sentence to Output Token

---

## Overview

This document walks through every step an input sentence takes inside a large language model (GPT-2 style), from raw text all the way to a predicted next token. A concrete example is shown at every stage.

**Model specs used throughout:** embedding dimension = 768, 12 attention heads, context length = 1024, vocabulary size = 50,257, 12 transformer blocks.

> **Note on example numbers:** The real embedding dimension is 768. Writing 768 numbers is not practical, so all vector examples show **4 values** and mark them with `[...]` to mean "768 values in reality." The logic is identical — only the size differs.

---

## Running Example

**Input sentence:** `"The cat sat on"`

This sentence has 4 words, which conveniently gives us 4 tokens for this example.

---

## Stage 1 — Tokenization

**What happens:** The input sentence is broken into smaller units called **tokens** using **Byte Pair Encoding (BPE)**. Each token is then converted to an integer **token ID** from the model's vocabulary (50,257 unique tokens for GPT-2).

- BPE is a learned algorithm — it merges frequently occurring character pairs into single tokens.
- Common words like "The" and "cat" each become one token. Rare words may be split into multiple tokens (e.g., "unhappiness" → "un", "happiness").

> **Context length constraint:**
> - **During training:** long documents are split into chunks of at most 1024 tokens *before* they reach the model (at the data preparation stage). The model therefore never encounters a sequence longer than the context length during training — this is handled offline, not at runtime.
> - **During inference:** the user controls the input and it can be any length. If the prompt exceeds 1024 tokens, a **sliding window** is applied at runtime — only the **last 1024 tokens** are kept and fed to the model. Earlier tokens are dropped from the current view.

> **Example:**
> ```
> Input sentence :  "The cat sat on"
>
> After BPE      :  ["The",  "cat",  "sat",  "on"]
>
> Token IDs      :  [ 464,   3797,   3332,   319 ]
>                     ↑        ↑       ↑       ↑
>                   "The"   "cat"   "sat"   "on"
>                   is ID   is ID   is ID   is ID
>                    464    3797    3332     319
>                   in the vocabulary of 50,257 tokens
> ```
> We now have **4 token IDs**: [464, 3797, 3332, 319]

---

## Stage 2 — Embedding Lookup

The model cannot work with raw integers. Each token ID must be converted into a rich numerical vector that captures meaning.

---

### 2a. Token Embeddings

**What happens:** Each token ID is used as a row index to look up a vector from the **Token Embedding Table**.

- Table size: **(50,257 rows × 768 columns)** — the number of rows is **always equal to the vocabulary size** (one row per token). If the vocabulary has 50,257 tokens, the table has exactly 50,257 rows.
- This table is learned during training — it is not hand-crafted.

> **Example:**
> ```
> Token Embedding Table  (50,257 × 768):
>
>   Row 0    :  [ 0.10, -0.22,  0.05,  0.88, ...]   ← token ID 0
>   Row 1    :  [-0.31,  0.14,  0.72, -0.19, ...]   ← token ID 1
>   ...
>   Row 319  :  [ 0.55, -0.11,  0.30,  0.44, ...]   ← "on"   (ID 319)
>   ...
>   Row 464  :  [ 0.21, -0.45,  0.83,  0.12, ...]   ← "The"  (ID 464)
>   ...
>   Row 3332 :  [-0.60,  0.88, -0.14,  0.27, ...]   ← "sat"  (ID 3332)
>   ...
>   Row 3797 :  [ 0.33,  0.05, -0.72,  0.61, ...]   ← "cat"  (ID 3797)
>   ...
>
> We look up rows 464, 3797, 3332, 319 and stack them:
>
>   Token Embeddings (4 × 768):
>   "The" → [ 0.21, -0.45,  0.83,  0.12, ...]
>   "cat" → [ 0.33,  0.05, -0.72,  0.61, ...]
>   "sat" → [-0.60,  0.88, -0.14,  0.27, ...]
>   "on"  → [ 0.55, -0.11,  0.30,  0.44, ...]
> ```

---

### 2b. Positional Embeddings

**What happens:** The model processes all tokens simultaneously — it has no natural sense of order. To tell it "this token is first, that one is third," we add a **positional embedding** for each position.

- **Positional Embedding Table** size: **(1024 rows × 768 columns)** — the number of rows is **always equal to the context length** (one row per position). If the context length is 1024, the table has exactly 1024 rows.
- This table is also learned during training.
- Token at position 0 gets positional vector 0, token at position 3 gets positional vector 3.

> **Example:**
> ```
> Positional Embedding Table  (1024 × 768):
>
>   Row 0  :  [ 0.00,  0.01,  1.00,  0.01, ...]   ← position 0 ("The")
>   Row 1  :  [ 0.84,  0.54, -0.46,  0.99, ...]   ← position 1 ("cat")
>   Row 2  :  [ 0.91, -0.42,  0.40, -0.65, ...]   ← position 2 ("sat")
>   Row 3  :  [ 0.14,  0.99,  0.14, -0.99, ...]   ← position 3 ("on")
> ```

---

### 2c. Combined Embedding

**What happens:** For each token, its token embedding and positional embedding are **added together** element by element. This gives one vector per token that encodes both *what* the token is and *where* it sits in the sentence.

> **Example:**
> ```
> Combined = Token Embedding + Positional Embedding
>
> "The" (pos 0):
>   [ 0.21, -0.45,  0.83,  0.12, ...]   ← token embedding
> + [ 0.00,  0.01,  1.00,  0.01, ...]   ← positional embedding
> = [ 0.21, -0.44,  1.83,  0.13, ...]   ← combined embedding ✓
>
> "cat" (pos 1):
>   [ 0.33,  0.05, -0.72,  0.61, ...]
> + [ 0.84,  0.54, -0.46,  0.99, ...]
> = [ 1.17,  0.59, -1.18,  1.60, ...]   ← combined embedding ✓
>
> (same for "sat" and "on")
>
> Final Combined Embedding matrix shape:  (4, 768)
>   Row 0 → "The"
>   Row 1 → "cat"
>   Row 2 → "sat"
>   Row 3 → "on"
> ```

---

## Stage 3 — Dropout (After Embeddings)

**What happens:** The (4, 768) matrix is passed through a **Dropout layer**. Some values in the matrix are randomly set to zero. This is a regularization technique that prevents the model from becoming too dependent on any single value.

- Each element is independently zeroed with probability `p` (e.g., dropout rate = 0.1 means each value has a 10% chance of being zeroed).
- **Which** values are zeroed changes every forward pass — but the **count** also varies. A 10% rate on 100 values means *on average* 10 are zeroed, but one pass might zero 8, another 13, another 10. Think of it like flipping 100 coins each with a 10% chance of heads — you will not always get exactly 10 heads.
- **Dropout is active during pre-training.** It is switched off in two situations:
  - **During inference** — the model is generating predictions, not learning.
  - **During fine-tuning** — in both classification and instruction fine-tuning, `drop_rate=0` is set in the config, effectively disabling dropout. This is common practice when fine-tuning on smaller datasets, where dropping activations can hurt the learning signal.

> **Important clarifications:**
> - Dropout zeros **activation values** — meaning the numbers inside the (4, 768) matrix that is currently flowing through the network. These are not the model's learned parameters; they are the intermediate results produced by the embedding lookup for this particular input.
> - The **weight parameters** (the embedding tables, W_Q, W_K, W_V, W_O, FFN weight matrices, etc.) are completely separate and are **never touched by dropout**. They are only updated by backpropagation.
> - The **weights are still trained** across iterations — they just receive zero gradient for the elements that were dropped in that particular pass.
> - The count of zeroed values is not fixed — it is the *average* that equals `dropout_rate × total_elements`, not the exact count every time.

> **Example** (dropout rate = 0.1 on the full 4×768 = 3072 values):
> ```
> Forward pass 1: 287 values zeroed  (≈10%, scattered randomly across all rows)
> Forward pass 2: 301 values zeroed  (≈10%, a completely different random set)
> Forward pass 3: 294 values zeroed  (≈10%, different again)
>
> Which rows are affected varies too — pass 1 might zero more values from
> rows 0 and 1, pass 2 might zero more from rows 2 and 3. No pattern is fixed.
>
> Zooming into one row for "The" (showing 8 values):
>
>   Before dropout:
>   "The" → [ 0.21, -0.44,  1.83,  0.13,  0.67, -0.22,  0.55,  0.39, ...]
>
>   After dropout, pass 1 (positions 2 and 5 zeroed):
>   "The" → [ 0.21, -0.44,  0.00,  0.13,  0.67,  0.00,  0.55,  0.39, ...]
>
>   After dropout, pass 2 (positions 0 and 6 zeroed — different positions):
>   "The" → [ 0.00, -0.44,  1.83,  0.13,  0.67, -0.22,  0.00,  0.39, ...]
>
> The weights behind zeroed positions are NOT frozen — they get updated
> in other passes where they are not dropped.
> ```

---

## Stage 4 — Transformer Block (Repeated 12 Times)

The following steps form one **Transformer Block**. This block is stacked 12 times in GPT-2. Each block refines the representation of every token.

---

### 4a. Layer Normalization (Pre-Attention)

**What happens:** Before entering attention, each token's 768-dimensional vector is **normalized independently** (row by row). This means its values are rescaled to have mean ≈ 0 and standard deviation ≈ 1.

- **Why:** Prevents values from growing too large or too small as they pass through many layers. Keeps training stable.
- After normalization, two learnable parameters (scale γ and shift β) are applied so the model can adjust the normalized values.

> **Example** (showing 4 values of the "sat" token row):
> ```
> Before Layer Norm:
>   "sat" row → [ 2.40,  0.10, -1.20,  3.90, ...]
>
> Step 1 — compute mean and std of this row:
>   mean = (2.40 + 0.10 + (-1.20) + 3.90) / 4 = 1.30
>   std  ≈ 1.88
>
> Step 2 — normalize each value:
>   (2.40 - 1.30) / 1.88 =  0.59
>   (0.10 - 1.30) / 1.88 = -0.64
>   (-1.20 - 1.30) / 1.88 = -1.33
>   (3.90 - 1.30) / 1.88 =  1.38
>
> After Layer Norm:
>   "sat" row → [ 0.59, -0.64, -1.33,  1.38, ...]
>                ↑ mean ≈ 0, std ≈ 1 across the full 768 values
>
> This is done separately for every token row — "The", "cat", "sat", "on" each
> get their own normalization.
> ```

---

### 4b. Multi-Head Self-Attention

This is the core of the Transformer. It lets each token "look at" the other tokens and decide how much attention to give each one.

---

#### Step 1 — Compute Q, K, V

**What happens:** Three trainable weight matrices **W_Q, W_K, W_V** (each 768×768) transform the input into three new matrices: **Query (Q)**, **Key (K)**, and **Value (V)**.

- Think of it like a library system:
  - **Query** = what this token is *looking for*
  - **Key** = what each token *advertises about itself*
  - **Value** = the actual *content* each token will contribute

> **Example:**
> ```
> Input matrix X: (4, 768)
>
> X × W_Q  →  Q: (4, 768)    ← each token's "question"
> X × W_K  →  K: (4, 768)    ← each token's "label"
> X × W_V  →  V: (4, 768)    ← each token's "content"
>
> (The actual numbers in Q, K, V depend on the learned weight matrices W_Q, W_K, W_V)
> ```

---

#### Step 2 — Split into 12 Heads

**What happens:** Each of Q, K, V is split into 12 equal slices along the 768-dimension axis. Each slice (head) independently learns a different type of relationship between tokens.

- 768 ÷ 12 = **64 dimensions per head**
- Head 1 might learn grammar relationships, Head 2 might learn subject-verb agreement, etc.

> **Example:**
> ```
> Q: (4, 768)  →  split into 12 heads
>
>   Head 1 Q  (4, 64): columns   0– 63 of Q
>   Head 2 Q  (4, 64): columns  64–127 of Q
>   Head 3 Q  (4, 64): columns 128–191 of Q
>   ...
>   Head 12 Q (4, 64): columns 704–767 of Q
>
> Same split applied to K and V.
> We now have 12 independent groups: (Q1,K1,V1), (Q2,K2,V2) ... (Q12,K12,V12)
> ```

---

#### Step 3 — Attention Scores

**What happens:** For each head, we compute how much every token should attend to every other token by taking the **dot product** of Q with the transpose of K, then **scaling** the result.

- (4, 64) × (64, 4) → **(4, 4)** score matrix
- Scale by dividing by √64 = 8 — prevents dot products from getting so large that softmax gradients vanish.
- Cell (i, j) = "how relevant is token j to token i?"

> **Example** (Head 1 scores, simplified to 4 tokens):
> ```
> Attention Scores (after ÷8 scaling, before masking), Head 1:
>
>              "The"  "cat"  "sat"  "on"
>   "The"  → [  1.2,   0.8,   0.3,  0.1 ]
>   "cat"  → [  0.9,   2.1,   0.6,  0.2 ]
>   "sat"  → [  0.4,   1.8,   2.5,  0.3 ]
>   "on"   → [  0.2,   0.5,   1.1,  1.9 ]
>
> These numbers are already divided by √64 = 8.
> Without scaling, the raw dot products would be much larger, causing
> softmax to produce near-zero gradients and making training unstable.
>
> Reading row "sat": token "sat" finds "sat" itself most relevant (2.5),
> then "cat" (1.8), then "The" (0.4), then "on" (0.3).
> ```

---

#### Step 4 — Causal Masking

**What happens:** This is a language *generation* model — when predicting token 2, it must not be allowed to peek at tokens 3 or 4 (they don't exist yet at generation time). Causal masking enforces this by replacing future positions with **−∞** before softmax.

- Softmax of −∞ = 0, so those positions contribute nothing.

> **Example:**
> ```
> After applying causal mask (upper triangle → -∞):
>
>              "The"   "cat"   "sat"   "on"
>   "The"  → [  1.2,    -∞,     -∞,    -∞  ]  ← can only see itself
>   "cat"  → [  0.9,   2.1,     -∞,    -∞  ]  ← sees "The" and "cat"
>   "sat"  → [  0.4,   1.8,    2.5,    -∞  ]  ← sees "The", "cat", "sat"
>   "on"   → [  0.2,   0.5,    1.1,   1.9  ]  ← sees all four tokens
>
> "The" (first token) can only attend to itself.
> "on"  (last token) can attend to all — it has the full context.
> ```

---

#### Step 5 — Attention Weights (Softmax)

**What happens:** Softmax converts each row into **probabilities** that sum to 1. The −∞ positions become 0.

> **Example:**
> ```
> After Softmax (Head 1):
>
>              "The"  "cat"  "sat"  "on"
>   "The"  → [ 1.00,  0.00,  0.00, 0.00 ]  ← 100% on itself
>   "cat"  → [ 0.27,  0.73,  0.00, 0.00 ]  ← 27% "The", 73% "cat"
>   "sat"  → [ 0.07,  0.33,  0.60, 0.00 ]  ← 7% "The", 33% "cat", 60% "sat"
>   "on"   → [ 0.06,  0.13,  0.27, 0.54 ]  ← spread across all four
>
> These are the attention weights. Each row sums to 1.
> "sat" pays the most attention to itself, then "cat", then "The" — this
> makes sense: to understand "sat", the subject "cat" is very relevant.
> ```

---

#### Step 6 — Dropout on Attention Weights

Dropout is applied to the (4, 4) attention weight matrix — some attention connections are randomly zeroed to prevent the model from always relying on the same patterns.

---

#### Step 7 — Weighted Sum with V (Context Matrix)

**What happens:** Each token's attention weights are used to take a **weighted blend** of the Value vectors. The result is a new vector for each token that now carries context from other tokens.

> **Example** (Head 1, focus on "sat"):
> ```
> Attention weights for "sat": [0.07, 0.33, 0.60, 0.00]
>               (meaning: 7% "The", 33% "cat", 60% "sat", 0% "on")
>
> Value vectors (V1), each of dim 64 (shown as 4 values):
>   V["The"] → [ 0.5, -0.3,  0.8,  0.2 ]
>   V["cat"] → [-0.4,  0.9, -0.1,  0.6 ]
>   V["sat"] → [ 0.7, -0.2,  0.5, -0.4 ]
>   V["on"]  → [ 0.1,  0.4,  0.3,  0.7 ]
>
> New "sat" vector = 0.07×V["The"] + 0.33×V["cat"] + 0.60×V["sat"] + 0.00×V["on"]
>                  = 0.07×[0.5,-0.3,0.8,0.2]
>                  + 0.33×[-0.4,0.9,-0.1,0.6]
>                  + 0.60×[0.7,-0.2,0.5,-0.4]
>                  + 0.00×[0.1,0.4,0.3,0.7]
>                  = [0.035,-0.021,0.056,0.014]
>                  + [-0.132,0.297,-0.033,0.198]
>                  + [0.420,-0.120,0.300,-0.240]
>                  + [0.000,0.000,0.000,0.000]
>                  = [ 0.32,  0.16,  0.32, -0.03 ]
>
> "sat" now has a context-aware vector — it carries information from
> "The" (7%), "cat" (33%), and itself (60%), giving it full context
> of all tokens seen so far.
> ```
> Shape of output per head: **(4, 64)**

---

#### Step 8 — Concatenate All Heads

**What happens:** The 12 context matrices (one per head) are joined side by side into one matrix.

> **Example:**
> ```
> Head 1  output: (4, 64)   ← learned one type of relationship
> Head 2  output: (4, 64)   ← learned another type
> ...
> Head 12 output: (4, 64)
>
> Concatenated horizontally → (4, 768)
>   [Head1 | Head2 | Head3 | ... | Head12]
>    64       64      64             64     = 768 total
> ```

---

#### Step 9 — Output Projection (W_O)

**What happens:** The concatenated (4, 768) matrix is multiplied by a trainable **output projection matrix W_O** of shape (768 × 768). This lets the model blend information across the 12 heads before moving forward.

> **Example:**
> ```
> (4, 768)  ×  W_O (768, 768)  →  (4, 768)
>
> Each token's vector is now a mix of all 12 heads' findings.
> ```

---

#### Step 10 — Dropout + Residual Connection (After Attention)

**What happens:**
1. Dropout is applied to the (4, 768) output.
2. The original input to this attention block is **added back** (residual / shortcut connection).

> **Example:**
> ```
> MHA output (after dropout):  [ 0.32,  0.16, -0.44,  0.09, ...]
> Original block input:        [ 0.59, -0.64, -1.33,  1.38, ...]   ← from LayerNorm
>
> After residual addition:     [ 0.91, -0.48, -1.77,  1.47, ...]
>
> Why residual? Without it, as data flows through 12 blocks, gradients can
> vanish or explode. Adding the original values back gives the gradient a
> "shortcut highway" to flow backward easily during training.
> ```

---

### 4c. Layer Normalization (Pre-FFN)

The same row-wise normalization from 4a is applied again — this time to stabilize values before the feed-forward network.

---

### 4d. Feed-Forward Network (FFN)

**What happens:** Each token's 768-dimensional vector is passed independently through a two-layer network that first **expands** it to 3072 dimensions, applies a non-linear activation, then **compresses** it back to 768.

- This is where the model stores and applies learned factual knowledge (e.g., "Paris is the capital of France").
- **GELU** (Gaussian Error Linear Unit): a smooth activation function — similar to ReLU but with a softer curve near zero.

> **Example** (following "sat" token through FFN):
> ```
> Input "sat" vector: (1, 768)
>
>   Linear Layer 1: (768 → 3072)
>   "sat" → (1, 3072) vector   ← expanded, more expressive space
>
>   GELU activation applied element-wise:
>   Negative values are gently suppressed; positives pass through.
>
>   Linear Layer 2: (3072 → 768)
>   "sat" → (1, 768) vector   ← compressed back to model dimension
>
> All 4 tokens go through FFN independently but in parallel.
> Output shape: (4, 768) — same as input.
> ```

#### Dropout + Residual Connection (After FFN)

Same pattern as after attention:
1. Dropout is applied to the FFN output.
2. The FFN's input is added back as a residual connection.

> **Example:**
> ```
> FFN output (after dropout):  [ 0.44, -0.81,  0.22,  1.05, ...]
> FFN input (pre-FFN vector):  [ 0.91, -0.48, -1.77,  1.47, ...]
>
> After residual addition:     [ 1.35, -1.29, -1.55,  2.52, ...]
>
> This completes ONE transformer block.
> ```

---

### This Entire Block Repeats 12 Times

Each of the 12 transformer blocks refines the token representations further.

> **Intuition:** Block 1 might learn basic syntax. Block 6 might capture semantic relationships. Block 12 might make fine-grained decisions about the next word. Each block builds on the previous one.

```
Block 1  input:  (4, 768)  ← raw combined embeddings
Block 1  output: (4, 768)  → fed into Block 2
Block 2  output: (4, 768)  → fed into Block 3
...
Block 12 output: (4, 768)  → fed to Final LayerNorm
```

---

## Stage 5 — Final Layer Normalization

After all 12 transformer blocks, one final row-wise Layer Normalization is applied to stabilize the (4, 768) matrix before the output layer.

---

## Stage 6 — Linear (Output) Layer

**What happens:** The (4, 768) matrix is multiplied by a weight matrix of shape (768 × 50,257). This projects every token's 768-dim vector into a **score for every word in the vocabulary**.

> **Example:**
> ```
> (4, 768)  ×  (768, 50,257)  →  (4, 50,257)
>
> Each row now has 50,257 raw scores (called logits).
>
> Row for "on" (last token), a few logits:
>   Score for "the"   = 3.21   ← high
>   Score for "a"     = 2.87
>   Score for "my"    = 1.44
>   Score for "mat"   = 3.68   ← highest
>   Score for "table" = 2.10
>   Score for "pizza" = -1.20  ← low (doesn't fit context)
>   ...
> ```

---

## Stage 7 — Softmax & Token Prediction

**What happens:** Softmax converts the raw logits into **probabilities** (each row sums to 1).

> **Example** (last token "on" row, a few entries after softmax):
> ```
>   "mat"   → 0.34  (34%) ← highest probability
>   "the"   → 0.21  (21%)
>   "a"     → 0.17  (17%)
>   "my"    → 0.09   (9%)
>   "table" → 0.07   (7%)
>   ...all others share remaining ~12%
> ```
> The model predicts **"mat"** as the next token after "The cat sat on".

The output is a **(4, 50,257)** matrix — one probability distribution per input token.

---

### Training vs. Inference — Which rows do we use?

#### During Training — Use ALL 4 tokens (all 4 rows)

We have the full correct sentence already, so we can learn from every token position at once.

> **Example:**
> ```
> Full training sentence: "The cat sat on the mat"
>
> Input  tokens: ["The",  "cat",  "sat",  "on" ]
> Target tokens: ["cat",  "sat",  "on",   "the"]
>                  ↑        ↑       ↑        ↑
>              Token 1  Token 2 Token 3  Token 4
>              ("The")  ("cat") ("sat")  ("on")
>              should   should  should   should
>              predict  predict predict  predict
>              "cat"    "sat"   "on"     "the"
>
> All 4 tokens' output rows are compared against their targets.
> Loss is computed for each token and averaged.
> → 1 forward pass teaches the model 4 token transitions at once.
> ```

#### During Inference — Use ONLY the last row

We generate one token at a time. The last token's row tells us what comes next.

> **Example:**
> ```
> Step 1 — Input: ["The", "cat", "sat", "on"]
>           Use last row ("on") → predicts "mat"
>           New sequence: ["The", "cat", "sat", "on", "mat"]
>
> Step 2 — Input: ["The", "cat", "sat", "on", "mat"]
>           Use last row ("mat") → predicts "."
>           New sequence: ["The", "cat", "sat", "on", "mat", "."]
>
> Step 3 — Input: ["The", "cat", "sat", "on", "mat", "."]
>           Use last row (".") → predicts end-of-sequence token → stop.
>
> Rows 1–3 outputs are computed but ignored at every inference step.
> Only the last row matters each time.
> ```

---

## Stage 8 — Loss & Backpropagation (Training Only)

**What happens:** The predicted probabilities are compared against the true next tokens using **cross-entropy loss**. Backpropagation then pushes gradients backward through every layer to update every trainable parameter.

> **Example:**
> ```
> Predictions vs Targets:
>
>   Row 1 → model predicted "cat" with prob 0.72 | target: "cat"  ✓  (low loss)
>   Row 2 → model predicted "sat" with prob 0.61 | target: "sat"  ✓  (low loss)
>   Row 3 → model predicted "at"  with prob 0.44 | target: "on"   ✗  (higher loss)
>   Row 4 → model predicted "the" with prob 0.35 | target: "the"  ✓  (low loss)
>
> Average cross-entropy loss is computed across all 4 rows.
>
> Backpropagation updates:
>   ✦ Token Embedding Table     (50,257 × 768)
>   ✦ Positional Embedding Table (1024  × 768)
>   ✦ W_Q, W_K, W_V, W_O in each of the 12 blocks
>   ✦ Layer normalization γ and β parameters
>   ✦ FFN weight matrices (all 12 blocks)
>
> Every weight is nudged slightly to make better predictions next time.
> ```

> **Teacher forcing:** During training, even if the model predicted the wrong token for Row 3, the correct token "on" is still fed as input in the next training step — not the model's wrong guess. This keeps training stable and fast.

---

### What Is the Model Actually Learning?

The training objective sounds deceptively simple: *predict the next token*. But to predict the next token well across millions of sentences, the model is forced to learn a remarkable amount:

- **Grammar and syntax** — "The cat ___" is far more likely to be followed by "sat" than "jumped the on" — so the model must learn sentence structure.
- **Facts about the world** — "The capital of France is ___" requires knowing that Paris follows, not London.
- **Context and meaning** — "I deposited money at the ___" points to "bank" (financial), while "I sat by the ___" points to "bank" (river). The model must learn to distinguish meaning from context.
- **Long-range dependencies** — "The trophy didn't fit in the suitcase because it was too ___" — the model must track what "it" refers to across the sentence to predict "big" or "large".

None of this is explicitly taught. The model is never told "this is a grammar rule" or "Paris is the capital of France." It discovers all of it purely by being trained to minimise the loss on next-token prediction across a massive corpus of text.

This is why scale matters: the more text the model is trained on, the richer and more nuanced the patterns it learns — and the more capable it becomes at generating coherent, knowledgeable, and contextually appropriate text.

---

## Full Workflow at a Glance

```
Input: "The cat sat on"
     ↓
Tokenization (BPE)
  "The"→464  "cat"→3797  "sat"→3332  "on"→319
     ↓
Token Embedding Lookup  (table: 50,257 × 768)
  464  → [0.21, -0.45,  0.83,  0.12, ...]
  3797 → [0.33,  0.05, -0.72,  0.61, ...]
  3332 → [-0.60, 0.88, -0.14,  0.27, ...]
  319  → [0.55, -0.11,  0.30,  0.44, ...]
     +
Positional Embedding Lookup  (table: 1024 × 768)
  pos0 → [0.00,  0.01,  1.00,  0.01, ...]
  pos1 → [0.84,  0.54, -0.46,  0.99, ...]
  pos2 → [0.91, -0.42,  0.40, -0.65, ...]
  pos3 → [0.14,  0.99,  0.14, -0.99, ...]
     ↓
Combined Embedding  →  shape: (4, 768)
     ↓
Dropout  (random ~10% values zeroed, training only)
     ↓
┌─────────────────────────────────────────────────────┐
│  Transformer Block × 12                             │
│                                                     │
│  LayerNorm  (normalize each token row independently)│
│     ↓                                               │
│  Multi-Head Self-Attention (12 heads)               │
│    compute Q, K, V  →  split into 12 heads          │
│    attention scores  →  scale (÷8)                  │
│    causal mask (-∞ for future)  →  softmax          │
│    → attention weights (4,4) per head               │
│    dropout  →  weighted sum with V  →  (4,64)/head  │
│    concatenate all heads  →  (4,768)                │
│    output projection W_O  →  (4,768)                │
│     ↓                                               │
│  Dropout + Residual Connection (add block input)    │
│     ↓                                               │
│  LayerNorm                                          │
│     ↓                                               │
│  Feed-Forward Network                               │
│    768 → 3072 (Linear + GELU) → 768 (Linear)       │
│     ↓                                               │
│  Dropout + Residual Connection (add FFN input)      │
└─────────────────────────────────────────────────────┘
     ↓
Final LayerNorm  →  (4, 768)
     ↓
Linear Layer  (768 → 50,257)  →  (4, 50,257) logits
     ↓
Softmax  →  (4, 50,257) probabilities
     ↓
┌──────────────────────────────────────────────────────────┐
│  TRAINING                       │  INFERENCE             │
│                                 │                        │
│  Use ALL 4 output rows          │  Use ONLY last row     │
│  (ground truth available for    │  ("on" → predicts next │
│   every position)               │   token in sequence)   │
│           ↓                     │           ↓            │
│  Row 1: predicted "cat"  ✓      │  Pick highest prob     │
│  Row 2: predicted "sat"  ✓      │  token → "mat"         │
│  Row 3: predicted "at"   ✗      │           ↓            │
│  Row 4: predicted "the"  ✓      │  Append "mat" to input │
│           ↓                     │  Re-run model          │
│  Cross-entropy loss              │  (repeat until done)   │
│           ↓                     │                        │
│  Backprop → update ALL weights  │                        │
│  (embeddings, W_Q/K/V/O,        │                        │
│   LayerNorm, FFN)               │                        │
└──────────────────────────────────────────────────────────┘
```

---

## Training vs. Inference — Key Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Output rows used | **All rows** (all 4 outputs) | **Last row only** |
| Next input | **Correct tokens** (teacher forcing) | **Model's own predicted token** |
| Dropout | **Active** during pre-training; **disabled** (drop_rate=0) during fine-tuning | **Disabled** (all values pass through) |
| Goal | Optimize weights via loss + backprop | Generate the next token, one at a time |

> **Why all rows in training but only the last in inference?**
> During training we know the full correct answer, so we squeeze 4 learning signals out of 1 forward pass. During inference we are generating new text one token at a time — the future does not exist yet, so only the most recent token's prediction is used.

---

## Key Numbers Summary (GPT-2 Base)

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 50,257 |
| Embedding dimension | 768 |
| Context length | 1024 |
| Attention heads | 12 |
| Dimension per head | 64 (= 768 ÷ 12) |
| Transformer blocks | 12 |
| FFN inner dimension | 3,072 (= 768 × 4) |
| Scale factor in attention | 8 (= √64) |

---

## Common Misconceptions to Avoid

| Misconception | Correct Understanding |
|---------------|----------------------|
| Dropout removes model weights | Dropout zeros **activation values**, not weight parameters |
| Dropped values are never trained | All weights are trained; dropped activations just get zero gradient for that one pass |
| A fixed number of values are dropped | Dropout is probabilistic — the count varies each forward pass |
| FFN maps 768 → 768 directly | FFN first expands to **3,072** (GELU), then compresses back to 768 |
| Multi-head output goes straight to residual | A **W_O projection** (768×768) is applied between concatenation and residual |
| Residual connections preserve original values | Their primary role is **gradient flow** through deep networks |
| During inference, all output rows are used | Only the **last token's output row** is used at each generation step |
| Teacher forcing means the model is not learning | The model still learns — correct inputs just make the gradient signal cleaner |

---

*Notes prepared for students of Dr. Navdeep Singh, CSE, Punjabi University, Patiala.*
