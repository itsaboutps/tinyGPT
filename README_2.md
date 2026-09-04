# TinyGPT Technical Onboarding

## Complete End-to-End Architecture, Data, Training, Checkpointing, and Generation Guide

**Project status:** Base TinyGPT lifecycle completed through Topic 20  
**Current phase:** Ready for Topic 21 — base-model evaluation and learning analysis  
**Development environment:** Windows 11, Python 3.14.4, PyTorch 2.13.0 CPU build, 32 GB RAM, Intel Core Ultra 7 265U  
**Primary purpose:** Learn how a GPT-style language model works by building the complete system from first principles at a scale that can run on a personal CPU-only laptop

---

# 1. Documentation scope and verification status

This document describes the complete TinyGPT system designed and implemented through the following stages:

1. Development environment and project structure
2. Text-data preparation
3. Character and byte tokenization
4. A custom byte-level BPE tokenizer
5. Tokenized dataset creation
6. Batch and target construction
7. Token embeddings
8. Rotary positional embeddings
9. Query, Key, and Value projections
10. Multi-head causal self-attention
11. RMSNorm
12. SwiGLU
13. Transformer blocks
14. A complete decoder-only TinyGPT
15. Cross-entropy loss
16. Backpropagation
17. AdamW optimization
18. Learning-rate scheduling
19. Validation and perplexity
20. Checkpointing and exact resume
21. Autoregressive text generation
22. Greedy, temperature, top-k, and top-p decoding

The project was built progressively, with a diagnostic script for nearly every component.

The directly uploaded and inspected source file was `scripts/train.py`. That file confirms that the training entrypoint composes the model configuration, token dataset, TinyGPT model, evaluator, optimizer, scheduler, training step, device utilities, parameter utilities, checkpoint manager, and hashing utilities. fileciteturn0file0L5-L39 fileciteturn0file0L44-L57

The uploaded version of `train.py` originally had checkpoint/resume control flow in the wrong place. A corrected version was produced, the stale experiment-directory conflict was resolved, and the user confirmed that issue was sorted. The descriptions below reflect the intended corrected architecture.

The other file descriptions reflect the implementations built during the project. They have not all been independently reopened from the user’s local laptop in one final repository-wide audit.

---

# 2. Purpose of the TinyGPT project

## 2.1 What the project is

TinyGPT is a small decoder-only Transformer language model built for learning.

It is designed to demonstrate the same major concepts found in much larger GPT-style systems:

```text
Text corpus
   ↓
Tokenizer training
   ↓
Token IDs
   ↓
Context windows
   ↓
Token embeddings
   ↓
Transformer blocks
   ↓
Vocabulary logits
   ↓
Cross-entropy loss
   ↓
Backpropagation
   ↓
Optimizer updates
   ↓
Checkpoint
   ↓
Autoregressive text generation
```

The model is intentionally small enough to run on a CPU-only laptop, while retaining the major architectural components of a modern GPT-like decoder:

```text
Byte-level BPE tokenizer
Token embeddings
RMSNorm
Multi-head causal self-attention
RoPE
Residual connections
SwiGLU MLP
Stacked Transformer blocks
Final RMSNorm
Weight-tied language-model head
Next-token cross-entropy
AdamW
Warmup and cosine learning-rate decay
Checkpointing
Autoregressive decoding
```

## 2.2 What the project is not

TinyGPT is not currently equivalent to ChatGPT.

The current model is a **base language model**. It predicts continuations based on patterns learned from the training corpus.

It has not yet received:

- Instruction-following fine-tuning
- System, user, and assistant role formatting
- Assistant-only loss masking
- Preference optimization
- RLHF
- DPO
- Safety alignment
- Tool use
- Retrieval
- Conversation-memory management
- A production inference server
- A KV cache
- Distributed training

The current model may learn to generate short story-like text because it is trained on a TinyStories subset. It is not yet trained to respond helpfully to questions or follow arbitrary instructions.

## 2.3 Educational objective

The objective is not merely to call a pretrained model:

```python
model = AutoModel.from_pretrained(...)
```

Instead, the project exposes the machinery hidden by high-level libraries.

The developer learns:

- Why text must become token IDs
- How a BPE vocabulary is learned
- Why token IDs need embeddings
- Why order information is needed
- How Query, Key, and Value are calculated
- Why attention scores have shape `[B, H, T, T]`
- Why future tokens must be causally masked
- How multiple heads are combined
- Why Transformer blocks contain both attention and MLP sublayers
- How cross-entropy measures prediction error
- How gradients travel backward
- What an optimizer actually updates
- How a model is saved and resumed
- How generated tokens are sampled one at a time

---

# 3. Current scale and configuration

The default learning-oriented model configuration is approximately:

```text
Target tokenizer vocabulary: 1024
Context length:              128 tokens
Model width:                 128
Attention heads:             4
Head dimension:              32
Transformer blocks:          4
SwiGLU hidden width:         384
RoPE base:                   10000
RMSNorm epsilon:             0.00001
```

Training configuration is approximately:

```text
Batch size:                  8
Maximum learning rate:       0.0003
Minimum learning rate:       0.00003
Weight decay:                0.1
Gradient clipping norm:      1.0
Warmup steps:                50
Maximum training steps:      1000
Log interval:                10
Evaluation interval:         100
Evaluation batches:          10
Checkpoint interval:         50
Random seed:                 42
```

The actual vocabulary size may be smaller than 1024 if the BPE trainer runs out of sufficiently frequent pairs before consuming its full merge budget.

The training script therefore does not blindly assume that the vocabulary is exactly 1024. It loads the trained tokenizer and constructs the model with:

```python
ModelConfig(
    vocab_size=tokenizer.vocab_size
)
```

The tokenizer is the source of truth for vocabulary size.

---

# 4. Essential tensor notation

The project uses several standard dimension names.

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `T` | Sequence length or number of token positions |
| `C` | Main model width, also called `d_model` |
| `H` | Number of attention heads |
| `D` | Dimension per attention head |
| `F` | Feed-forward or SwiGLU hidden dimension |
| `V` | Vocabulary size |
| `N` | Number of tokens in an entire stored split |

For the default model:

```text
B = 8
T = 128
C = 128
H = 4
D = C / H = 32
F = 384
V ≈ 1024
```

The most important shape transitions are:

```text
Token stream:
[N]

Training batch:
[B, T]

Token embeddings:
[B, T, C]

Fused QKV:
[B, T, 3C]

Each Q, K, V:
[B, T, C]

Q/K/V split into heads:
[B, H, T, D]

Attention scores:
[B, H, T, T]

Attention output before merging heads:
[B, H, T, D]

Merged attention output:
[B, T, C]

SwiGLU expanded hidden state:
[B, T, F]

Transformer output:
[B, T, C]

Vocabulary logits:
[B, T, V]

Flattened training logits:
[B × T, V]

Flattened targets:
[B × T]
```

---

# 5. Repository architecture

The intended project structure is:

```text
tinygpt/
│
├── .venv/
│
├── configs/
│
├── data/
│   ├── raw/
│   │   └── input.txt
│   │
│   ├── processed/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── metadata.json
│   │
│   ├── tokenizer/
│   │   └── tokenizer.json
│   │
│   └── tokens/
│       ├── train.pt
│       ├── val.pt
│       ├── test.pt
│       └── metadata.json
│
├── tinygpt/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   └── token_dataset.py
│   │
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── char_tokenizer.py
│   │   ├── byte_tokenizer.py
│   │   ├── bpe.py
│   │   └── bpe_trainer.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── norm.py
│   │   ├── mlp.py
│   │   ├── block.py
│   │   ├── transformer.py
│   │   └── gpt.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   ├── step.py
│   │   ├── schedule.py
│   │   ├── evaluate.py
│   │   └── checkpoint.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   ├── sampling.py
│   │   └── generate.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── device.py
│       ├── random.py
│       ├── parameters.py
│       └── hashing.py
│
├── scripts/
│   ├── check_environment.py
│   ├── check_torch.py
│   ├── tensor_basics.py
│   ├── autograd_basics.py
│   ├── check_config.py
│   ├── check_seed.py
│   ├── check_experiment.py
│   ├── download_tinystories.py
│   ├── prepare_data.py
│   ├── check_char_tokenizer.py
│   ├── check_bytes.py
│   ├── check_byte_tokenizer.py
│   ├── inspect_tokenization.py
│   ├── check_bpe.py
│   ├── train_tokenizer.py
│   ├── check_saved_tokenizer.py
│   ├── prepare_tokens.py
│   ├── check_dataset_sizes.py
│   ├── check_token_dataset.py
│   ├── check_batch.py
│   ├── check_embeddings.py
│   ├── check_embedding_gradients.py
│   ├── check_model_input.py
│   ├── check_positions.py
│   ├── check_rope_math.py
│   ├── check_rope.py
│   ├── check_qkv.py
│   ├── check_attention_head.py
│   ├── check_multihead_attention.py
│   ├── check_rmsnorm.py
│   ├── check_attention_residual.py
│   ├── check_silu.py
│   ├── check_swiglu.py
│   ├── check_mlp_independence.py
│   ├── check_mlp_residual.py
│   ├── check_transformer_block.py
│   ├── check_block_gradients.py
│   ├── check_transformer_stack.py
│   ├── check_stack_gradients.py
│   ├── check_tinygpt.py
│   ├── check_weight_tying.py
│   ├── check_full_forward.py
│   ├── check_model_causality.py
│   ├── check_cross_entropy.py
│   ├── check_tinygpt_loss.py
│   ├── check_language_gradients.py
│   ├── check_gradient_descent.py
│   ├── check_optimizer_groups.py
│   ├── check_optimizer_step.py
│   ├── check_overfit_batch.py
│   ├── check_lr_schedule.py
│   ├── check_checkpoint_roundtrip.py
│   ├── train.py
│   ├── generate_greedy.py
│   ├── generate.py
│   └── check_generation.py
│
├── checkpoints/
│   └── tinystories_5mb_v1/
│       ├── run_metadata.json
│       ├── latest.pt
│       └── best.pt
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 6. Why the repository is divided this way

The project deliberately separates responsibilities.

```text
tinygpt/tokenizer/
```

knows how text becomes token IDs and back.

```text
tinygpt/data/
```

knows how text and token streams are stored, split, loaded, and batched.

```text
tinygpt/model/
```

knows the neural-network architecture.

```text
tinygpt/training/
```

knows how to calculate loss, optimize parameters, evaluate, schedule learning rates, and save checkpoints.

```text
tinygpt/generation/
```

knows how to load a trained model and turn logits into generated tokens.

```text
tinygpt/utils/
```

contains infrastructure shared by multiple parts of the project.

```text
scripts/
```

contains executable entrypoints and diagnostic experiments.

This separation prevents a single file from becoming responsible for every concern.

For example, the attention implementation does not know:

- Where the dataset is stored
- Which checkpoint directory is used
- How the tokenizer was trained
- Which command starts training
- Whether generation uses top-k

It receives a tensor and performs attention.

Likewise, the tokenizer does not know:

- How many Transformer blocks exist
- Which optimizer is used
- Whether the model is on CPU or GPU
- How validation loss is calculated

This is one of the project’s main scalability principles:

> Separate model architecture, data storage, optimization, and runtime infrastructure.

---

# 7. Root-level files and folders

## 7.1 `.venv/`

This is the Python virtual environment created using:

```powershell
python -m venv .venv
```

It contains the isolated Python interpreter and installed packages for TinyGPT.

The environment prevents project dependencies from being mixed with unrelated global Python packages.

The environment was verified through:

```text
C:\Users\prashant.bi.singh\Desktop\PERSONAL\tinygpt\.venv\Scripts\python.exe
```

PyTorch was also verified as a CPU build:

```text
2.13.0+cpu
```

## 7.2 `configs/`

This directory is a future extension point for external configuration files such as:

```text
tiny.yaml
small.yaml
cpu_debug.yaml
```

At the current stage, configuration is implemented with Python dataclasses in:

```text
tinygpt/config.py
```

The directory exists so configuration can later move to YAML or JSON without mixing configuration files into model code.

## 7.3 `requirements.txt`

This file should record installable Python dependencies.

At minimum, the current project needs:

```text
torch
datasets
```

PyTorch provides:

- Tensors
- Neural-network layers
- Automatic differentiation
- Optimizers
- Model serialization

The Hugging Face `datasets` package is used only for streaming a manageable TinyStories subset.

A production-quality version should pin tested versions instead of leaving dependencies unversioned.

## 7.4 `.gitignore`

This file prevents generated or machine-specific content from entering Git.

Typical entries include:

```text
.venv/
__pycache__/
*.pyc
.vscode/
*.log
checkpoints/
data/processed/
data/tokens/
```

The important idea is that source code belongs in Git, while large generated artifacts such as model checkpoints and processed token tensors normally require separate artifact storage.

## 7.5 `README.md`

This is intended to become the short public-facing entrypoint for the repository.

It should eventually explain:

- What TinyGPT is
- Installation commands
- Dataset-preparation commands
- Training command
- Resume command
- Generation command
- Model limitations

The current onboarding document contains much more detail than a normal README.

## 7.6 `tests/`

This is reserved for automated tests, preferably using `pytest`.

At present, many correctness checks exist as executable scripts under `scripts/`.

A future cleanup should move stable assertions from those scripts into formal tests.

---

# 8. Configuration system: `tinygpt/config.py`

This file centralizes parameters that control architecture, data, optimization, tokenization, and run behavior.

It prevents hardcoded values from being scattered across model files.

## 8.1 `ModelConfig`

`ModelConfig` describes the neural-network architecture.

Main fields:

```text
vocab_size
context_length
d_model
n_heads
n_layers
d_ff
rope_base
rms_norm_eps
```

### `vocab_size`

The number of token IDs the model can represent and predict.

This controls:

```text
Embedding table rows
LM-head output classes
Maximum legal token ID
```

The real value comes from the trained tokenizer.

### `context_length`

The maximum number of tokens the model can process at once.

Current value:

```text
128
```

The causal attention mask is created with this maximum size and sliced for shorter sequences.

### `d_model`

The main hidden width of each token representation.

Current value:

```text
128
```

After token embedding, each token is represented by 128 floating-point features.

### `n_heads`

The number of parallel attention heads.

Current value:

```text
4
```

### `head_dim`

This is calculated, not independently configured:

```text
head_dim = d_model // n_heads
```

Current value:

```text
128 // 4 = 32
```

### `n_layers`

The number of stacked Transformer blocks.

Current value:

```text
4
```

Each block has the same architecture but independent parameters.

### `d_ff`

The hidden width inside the SwiGLU feed-forward network.

Current value:

```text
384
```

Each token temporarily expands:

```text
128 → 384 → 128
```

inside every block.

### `rope_base`

Controls the frequency schedule used by rotary position embeddings.

Current value:

```text
10000
```

### `rms_norm_eps`

A small number added inside RMSNorm to prevent division by zero.

Current value:

```text
0.00001
```

### Model validation

`ModelConfig.__post_init__()` rejects invalid architecture combinations.

Important invariants include:

```text
vocab_size > 0
context_length > 0
d_model > 0
n_heads > 0
d_model % n_heads == 0
head_dim % 2 == 0
rms_norm_eps > 0
```

`d_model` must be divisible by `n_heads` because every head needs an integer number of features.

`head_dim` must be even because RoPE rotates feature pairs.

## 8.2 `TrainingConfig`

This class describes the optimization process.

Fields include:

```text
batch_size
learning_rate
min_learning_rate
weight_decay
grad_clip_norm
warmup_steps
max_steps
log_interval
eval_interval
eval_batches
checkpoint_interval
seed
```

### `batch_size`

Number of independent context windows used in one optimizer step.

Current value:

```text
8
```

### `learning_rate`

Maximum learning rate reached after warmup:

```text
0.0003
```

### `min_learning_rate`

Final learning rate approached through cosine decay:

```text
0.00003
```

### `weight_decay`

Regularization applied by AdamW to matrix-like parameters:

```text
0.1
```

### `grad_clip_norm`

Maximum allowed global gradient norm after scaling:

```text
1.0
```

### `warmup_steps`

Number of initial steps over which the learning rate rises gradually:

```text
50
```

### `max_steps`

Maximum number of optimizer updates:

```text
1000
```

### Logging, evaluation, and checkpoint intervals

```text
log_interval        = 10
eval_interval       = 100
eval_batches        = 10
checkpoint_interval = 50
```

### `seed`

Base random seed:

```text
42
```

Different deterministic offsets are used for:

- Training batch sampling
- Training evaluation sampling
- Validation evaluation sampling

## 8.3 `DataConfig`

This describes source and processed data paths.

Typical fields:

```text
raw_path = data/raw/input.txt
processed_dir = data/processed
train_fraction = 0.90
val_fraction = 0.05
test_fraction = 0.05
```

It validates that the fractions sum approximately to 1.0 using `math.isclose()`.

## 8.4 `TokenizerConfig`

This describes tokenizer training.

Typical fields:

```text
vocab_size = 1024
min_pair_frequency = 2
output_path = data/tokenizer/tokenizer.json
```

A byte-level tokenizer needs at least:

```text
256 byte tokens
+ at least one special token
```

The configuration therefore rejects impossibly small vocabulary targets.

## 8.5 `RunConfig`

This describes the experiment’s operational identity.

Fields:

```text
experiment_name
checkpoint_root
resume_from
```

Example:

```text
experiment_name = tinystories_5mb_v1
checkpoint_root = checkpoints
resume_from = None
```

A fresh run with this configuration writes to:

```text
checkpoints/tinystories_5mb_v1/
```

To continue the same run:

```text
resume_from =
checkpoints/tinystories_5mb_v1/latest.pt
```

---

# 9. Utility package: `tinygpt/utils/`

## 9.1 `device.py`

This file contains `get_device()`.

It selects the best supported runtime device:

```text
CUDA GPU, if available
MPS, if available on Apple systems
CPU otherwise
```

On the current Windows laptop:

```text
device = cpu
```

The purpose is to avoid hardcoding `"cpu"` throughout model files.

Correct device separation looks like:

```text
Model architecture
does not decide hardware.

Runtime code
selects the device.
```

The same model can later be moved to a GPU with:

```python
model.to(device)
```

without rewriting attention, MLP, or Transformer code.

## 9.2 `random.py`

This file contains `set_seed(seed)`.

It seeds:

```text
Python random generator
PyTorch global random generator
CUDA generators, when available
```

Randomness is used in:

- Weight initialization
- Training-window sampling
- Text-generation sampling
- Any future dropout

A fixed seed improves reproducibility.

A seed does not make every result universally bit-identical across all hardware and software combinations, but it greatly improves repeatability within the same environment.

## 9.3 `parameters.py`

This file contains utilities such as:

```text
count_parameters()
count_trainable_parameters()
parameter_size_bytes()
parameter_size_mb()
```

`count_parameters()` sums `parameter.numel()` over model parameters.

`count_trainable_parameters()` includes only parameters whose:

```text
requires_grad == True
```

`parameter_size_mb()` estimates memory occupied by parameter tensors themselves.

It does not include:

- Gradients
- Optimizer moments
- Activations
- Attention matrices
- Temporary tensors
- Dataset memory

## 9.4 `hashing.py`

This file contains `file_sha256(path)`.

It reads a file in chunks and calculates a SHA-256 fingerprint.

Fingerprints are used to establish data and tokenizer lineage.

For example:

```text
tokenizer.json → hash A
data/tokens/metadata.json → hash B
```

These hashes are saved inside training checkpoints.

Resume refuses to continue if the current tokenizer or token metadata does not match the checkpoint.

---

# 10. Data package: `tinygpt/data/`

## 10.1 `text.py`

This file manages plain-text corpus preparation.

### `load_text(path)`

Responsibilities:

1. Convert the string path into a `Path` object
2. Verify that the file exists
3. Read UTF-8 text
4. Reject empty or whitespace-only files
5. Return the full string

UTF-8 is explicitly used so the pipeline can represent:

```text
English
Hindi
accented text
emoji
other Unicode content
```

### `normalize_text(text)`

The normalization is intentionally conservative.

It normalizes line endings:

```text
Windows \r\n → \n
old-style \r  → \n
```

It removes null bytes:

```text
\x00
```

It does not:

- Lowercase the text
- Remove punctuation
- Remove spaces
- Remove newlines
- Stem words
- Strip all formatting

Case, punctuation, and formatting are useful language-model signals.

### `split_text()`

This divides one continuous corpus into train, validation, and test regions.

With 90/5/5:

```text
First 90%  → train
Next 5%    → validation
Final 5%   → test
```

This preserves local sequence continuity better than assigning individual characters randomly.

The current implementation is character-position based.

A future improvement should split TinyStories at document boundaries before concatenation so one story is never cut between train and validation.

### `save_text()`

Creates parent directories if needed and writes UTF-8 text.

### `text_sha256()`

Calculates a fingerprint of normalized corpus content.

This is stored in processed-data metadata.

## 10.2 `token_dataset.py`

This file wraps a stored one-dimensional token tensor.

### Construction

`TokenDataset(token_path)`:

1. Verifies that the file exists
2. Loads the tensor onto CPU
3. Uses `weights_only=True` because the file is expected to contain a tensor
4. Checks that the tensor is one-dimensional
5. Checks that its dtype is `torch.long`

A valid stored split looks like:

```text
tensor([318, 72, 801, 44, ...])
```

Shape:

```text
[N]
```

where `N` is the number of tokens in that split.

### `__len__()`

Returns the number of tokens.

### `get_window(start, context_length)`

Builds one language-model training example.

Given:

```text
tokens =
[10, 20, 30, 40, 50, 60]
```

with:

```text
start = 1
context_length = 3
```

it returns:

```text
x = [20, 30, 40]
y = [30, 40, 50]
```

The target is shifted by one token.

This teaches:

```text
20 → 30
20,30 → 40
20,30,40 → 50
```

after causal masking is applied inside the model.

### `get_batch()`

Inputs include:

```text
batch_size
context_length
device
optional torch.Generator
```

It:

1. Calculates the maximum legal starting position
2. Samples random starting positions
3. Builds an input and target window for each
4. Stacks windows into batches
5. Moves the batch to the requested device

Output shapes:

```text
x: [B, T]
y: [B, T]
```

With current defaults:

```text
x: [8, 128]
y: [8, 128]
```

The optional generator allows training, validation, and testing to use independent reproducible random streams.

---

# 11. Data artifacts

## 11.1 `data/raw/input.txt`

This is the raw source corpus.

Initially it contained a tiny hand-written text sample for pipeline testing.

It was later replaced with an approximately 5 MiB TinyStories subset downloaded through streaming.

Each story is written with a blank line separating it from the next story.

Conceptually:

```text
Story 1...

Story 2...

Story 3...
```

## 11.2 `data/processed/train.txt`

Contains the training portion of normalized raw text.

This is the only split used to:

- Train tokenizer merge rules
- Sample weight-updating training batches

## 11.3 `data/processed/val.txt`

Contains validation text.

It is used to measure unseen loss during model development.

Validation does not update model parameters.

The earlier error:

```text
Validation dataset is too small for configured context length
```

occurred because this split did not contain at least:

```text
context_length + 1
```

tokens.

Replacing the tiny corpus with a 5 MiB TinyStories subset solved the problem properly.

## 11.4 `data/processed/test.txt`

Contains held-out test text.

It is intentionally not used during normal training.

It should be used later for final evaluation after development decisions have been made using validation data.

## 11.5 `data/processed/metadata.json`

Records processed-corpus details such as:

```text
source path
SHA-256 fingerprint
total characters
train characters
validation characters
test characters
split fractions
```

This makes preprocessing traceable.

## 11.6 `data/tokenizer/tokenizer.json`

This is the trained tokenizer artifact.

It contains:

```text
format version
tokenizer type
EOS string
EOS token ID
ordered BPE merge rules
```

It defines the meaning of every token ID above the base byte range.

This file must remain paired with:

- Tokenized `.pt` files
- Model checkpoints
- Generated text

Retraining the tokenizer can change token meanings even when vocabulary size stays the same.

## 11.7 `data/tokens/train.pt`

A one-dimensional `torch.long` tensor containing encoded training tokens.

Stories are encoded separately and EOS is appended after each story.

Conceptually:

```text
Story 1 token IDs
EOS
Story 2 token IDs
EOS
Story 3 token IDs
EOS
```

## 11.8 `data/tokens/val.pt`

Validation token stream.

It is loaded by `TokenDataset` and sampled only during evaluation.

## 11.9 `data/tokens/test.pt`

Held-out test token stream.

## 11.10 `data/tokens/metadata.json`

Records:

```text
tokenizer path
tokenizer SHA-256
actual vocabulary size
character counts
UTF-8 byte counts
token counts by split
```

This file is hashed and linked to checkpoints.

---

# 12. Tokenizer package: `tinygpt/tokenizer/`

## 12.1 `char_tokenizer.py`

This is the simplest educational tokenizer.

### Vocabulary creation

It takes all unique characters from training text:

```python
sorted(set(text))
```

For:

```text
banana
```

the vocabulary might be:

```text
a
b
n
```

### Maps

It creates:

```text
stoi: string/character → integer
itos: integer → string/character
```

Example:

```text
a → 0
b → 1
n → 2
```

### Encoding

```text
banana
↓
[1, 0, 2, 0, 2, 0]
```

### Decoding

```text
[1, 0, 2, 0, 2, 0]
↓
banana
```

### Limitation

A character not present in the training vocabulary produces an out-of-vocabulary error.

The character tokenizer exists mainly to teach the tokenization contract:

```text
encode(text) → list[int]
decode(list[int]) → text
```

It is not the final tokenizer.

## 12.2 `byte_tokenizer.py`

This tokenizer maps UTF-8 bytes directly to token IDs.

Vocabulary size is fixed:

```text
256
```

A byte can have values from 0 through 255.

Example:

```text
hello
↓ UTF-8
[104, 101, 108, 108, 111]
```

Unicode characters may require several bytes.

An emoji can become several token IDs.

### Advantage

Any valid UTF-8 text can be represented without an unknown-token problem.

### Limitation

Sequences are inefficiently long because every byte is a token.

## 12.3 `bpe.py`

This file contains the reusable byte-level BPE tokenizer and merge mechanics.

### `MergeRule`

An immutable dataclass containing:

```text
left token ID
right token ID
new merged token ID
```

Example:

```text
97 + 110 → 256
```

If byte 97 is `a` and byte 110 is `n`, token 256 represents the byte sequence `an`.

The object is frozen so merge rules cannot be accidentally modified after creation.

### `merge_pair()`

Scans a token sequence left to right.

Whenever the configured adjacent pair occurs, it replaces both IDs with the new merged ID.

Example:

```text
Input:
[1, 2, 1, 2, 3]

Merge:
(1,2) → 10

Output:
[10, 10, 3]
```

The index advances by two after a merge and by one otherwise.

### `BPETokenizer.__init__()`

The tokenizer receives:

```text
ordered merge rules
EOS token ID
```

It builds several internal mappings.

#### `merge_to_id`

Maps:

```text
(left, right) → new token ID
```

#### `merge_rank`

Maps each merge pair to its training order.

BPE merge order matters because later merges can depend on earlier merged tokens.

#### `token_bytes`

Maps every token ID to the bytes it represents.

Base tokens:

```text
0–255 → one byte each
```

Learned tokens are reconstructed recursively.

If:

```text
97 + 110 → 256
```

then:

```text
token_bytes[256] = b"an"
```

### Encoding

Encoding starts with raw UTF-8 bytes.

Then, while a known merge is available:

1. Inspect adjacent token pairs
2. Find the available pair with the best, meaning earliest, merge rank
3. Replace all occurrences using `merge_pair()`
4. Repeat until no learned merge applies
5. Optionally append EOS

Output:

```text
text → list[int]
```

### Decoding

Decoding:

1. Converts normal token IDs back into their byte sequences
2. Buffers the bytes
3. Handles EOS separately
4. Decodes the combined byte sequence as UTF-8

Tokens are not decoded independently because one token may contain only part of a multibyte Unicode character.

### `save()`

Writes a JSON artifact containing merge rules and EOS metadata.

### `load()`

Reads the JSON file, reconstructs `MergeRule` objects, and creates a working tokenizer.

## 12.4 `bpe_trainer.py`

This file learns merge rules from representative training text.

### `count_pairs()`

Given:

```text
[10, 20, 30, 40]
```

it counts:

```text
(10,20)
(20,30)
(30,40)
```

A `Counter` records pair frequencies.

### `BPETrainer`

Main configuration:

```text
target vocabulary size
minimum pair frequency
```

Base vocabulary:

```text
256 bytes
```

One additional token is reserved for EOS.

Therefore the merge budget is approximately:

```text
target_vocab_size - 256 - 1
```

### Training algorithm

1. Convert training text to UTF-8 byte IDs
2. Count adjacent pairs
3. Choose the most frequent pair
4. Resolve ties deterministically using token IDs
5. Reject the pair if frequency is below the configured minimum
6. Assign a new token ID
7. Replace all occurrences
8. Save the merge rule
9. Repeat until the merge budget is exhausted or no useful pair remains
10. Assign the next available ID to EOS
11. Construct `BPETokenizer`

This is a transparent educational BPE implementation.

It is not optimized for multi-gigabyte tokenizer training.

For the current project, tokenizer training uses a representative subset of approximately 500,000 training characters rather than the entire 5 MiB corpus.

## 12.5 `tokenizer/__init__.py`

Exports selected tokenizer classes so other modules can use imports such as:

```python
from tinygpt.tokenizer import ByteTokenizer
```

rather than depending on internal file locations.

---

# 13. Why byte-level BPE was chosen

Character tokenization has a small vocabulary but long sequences and an unknown-character risk.

Word tokenization can create an enormous vocabulary and cannot naturally represent unseen words.

Raw byte tokenization has no unknown-token problem but uses too many tokens.

Byte-level BPE combines:

```text
Byte-level universality
+
learned subword compression
```

The base 256 tokens guarantee that any UTF-8 text can be represented.

Frequent byte sequences become larger tokens.

For example, repeated merges may conceptually learn:

```text
t + h → th
th + e → the
```

The text `the` may then require one token rather than three bytes.

A useful efficiency metric is:

```text
bytes per token =
number of UTF-8 bytes / number of BPE tokens
```

Raw byte tokenization is near one byte per token.

A trained BPE tokenizer should improve this value.

---

# 14. Model package: `tinygpt/model/`

## 14.1 `embeddings.py`

### `TokenEmbedding`

This class inherits from `nn.Module`.

It wraps:

```python
nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model
)
```

The weight matrix has shape:

```text
[V, C]
```

For a vocabulary of 1024 and model width 128:

```text
[1024, 128]
```

Each row is a trainable vector for one token ID.

Input:

```text
[B, T]
```

Output:

```text
[B, T, C]
```

Example:

```text
[8,128]
↓
[8,128,128]
```

Token IDs are arbitrary lookup keys. Token 500 is not numerically “greater in meaning” than token 100.

The embedding table gives each token a learnable continuous representation.

### `LearnedPositionEmbedding`

This was created as an educational comparison.

It contains an embedding table:

```text
[context_length, d_model]
```

Position IDs are generated with:

```text
0, 1, 2, ..., T-1
```

These position vectors can be added to token embeddings using broadcasting.

The final model does not use this class because it uses RoPE inside attention.

## 14.2 `rope.py`

This file implements Rotary Position Embeddings.

### Why position is necessary

Token embeddings tell the model what token appears.

They do not independently provide enough information about token order.

These sentences use the same token identities but have different meanings:

```text
dog bites man
man bites dog
```

### RoPE strategy

RoPE does not add a position vector directly to the token embedding.

It rotates Query and Key vectors according to token position.

Values are not rotated.

### Feature pairing

If:

```text
head_dim = 32
```

RoPE treats the dimensions as 16 pairs:

```text
(0,1)
(2,3)
...
(30,31)
```

Each pair behaves like a two-dimensional vector.

### Inverse frequencies

Different pairs rotate at different rates.

Conceptually:

```text
position × inverse frequency = angle
```

Cosine and sine of the angle are then used for rotation.

### Buffer

The inverse-frequency tensor is registered with:

```python
register_buffer(...)
```

It:

- Belongs to the model
- Moves with the model to CPU or GPU
- Is not optimized
- Is not a trainable parameter

### Input and output

RoPE expects:

```text
[B, H, T, D]
```

It returns the same shape.

It changes vector directions according to position but preserves vector magnitude approximately.

Position zero is unchanged because:

```text
cos(0) = 1
sin(0) = 0
```

## 14.3 `attention.py`

This file contains both an educational single attention head and the final multi-head implementation.

### `CausalSelfAttentionHead`

The educational version creates separate projection layers:

```text
Query projection
Key projection
Value projection
```

Input:

```text
[B,T,C]
```

Each projection outputs:

```text
[B,T,D]
```

RoPE is applied by temporarily adding a one-head dimension:

```text
[B,T,D]
↓
[B,1,T,D]
```

Attention scores are calculated as:

```text
Q @ Kᵀ
```

Shape:

```text
[B,T,D] @ [B,D,T]
→ [B,T,T]
```

Scores are scaled by:

```text
1 / sqrt(D)
```

A lower-triangular causal mask prevents access to future positions.

Softmax converts allowed scores into attention weights.

The output is:

```text
attention_weights @ V
```

Shape:

```text
[B,T,T] @ [B,T,D]
→ [B,T,D]
```

### `MultiHeadCausalSelfAttention`

This is the final attention layer used by Transformer blocks.

#### Fused QKV projection

Instead of three separate matrix multiplications, it uses:

```text
Linear(C → 3C)
```

Input:

```text
[B,T,C]
```

Fused result:

```text
[B,T,3C]
```

It is split into:

```text
Q: [B,T,C]
K: [B,T,C]
V: [B,T,C]
```

#### Splitting heads

Since:

```text
C = H × D
```

each tensor is reshaped:

```text
[B,T,C]
→ [B,T,H,D]
→ transpose
→ [B,H,T,D]
```

For current values:

```text
[8,128,128]
→ [8,128,4,32]
→ [8,4,128,32]
```

#### RoPE

RoPE is applied to Q and K:

```text
Q: [B,H,T,D]
K: [B,H,T,D]
```

V remains unchanged.

#### Scores

```text
Q @ Kᵀ
```

becomes:

```text
[B,H,T,D] @ [B,H,D,T]
→ [B,H,T,T]
```

Each batch item and each attention head gets its own token-to-token score matrix.

#### Scaling

Scores are multiplied by:

```text
D ** -0.5
```

This prevents dot-product magnitude from growing excessively with head dimension.

#### Causal mask

A Boolean lower-triangular matrix of shape:

```text
[context_length, context_length]
```

is stored as a buffer.

For a shorter sequence, it is sliced:

```text
mask[:T, :T]
```

Illegal future scores are replaced by:

```text
-inf
```

After softmax, their probability becomes zero.

#### Softmax

Applied over the final key-position dimension.

Every query row sums approximately to 1.

#### Weighted values

```text
attention_weights @ V
```

produces:

```text
[B,H,T,D]
```

#### Merge heads

The tensor is transposed:

```text
[B,H,T,D]
→ [B,T,H,D]
```

Then made contiguous and reshaped:

```text
[B,T,H,D]
→ [B,T,C]
```

#### Output projection

A final:

```text
Linear(C → C)
```

mixes information from the concatenated attention heads.

The final attention layer preserves shape:

```text
[B,T,C]
→ [B,T,C]
```

This enables residual addition.

## 14.4 `norm.py`

This file implements `RMSNorm`.

For each token vector, it computes:

\[
RMS(x)=\sqrt{\text{mean}(x^2)+\epsilon}
\]

Then:

\[
\hat{x}=x/RMS(x)
\]

Finally, it multiplies by a learned scale vector:

```text
weight: [C]
```

Input and output:

```text
[B,T,C]
```

The mean is taken over the final feature dimension with:

```text
keepdim=True
```

This produces:

```text
[B,T,1]
```

which broadcasts over all `C` features.

RMSNorm differs from LayerNorm because it does not subtract the mean.

Each RMSNorm layer has only:

```text
C
```

trainable parameters.

For `C=128`, that is 128 parameters.

## 14.5 `mlp.py`

This file implements the `SwiGLU` feed-forward network.

For every token independently:

```text
Input: [C]
```

Two separate projections are calculated:

```text
gate = SiLU(W_gate x)
up   = W_up x
```

Then:

```text
hidden = gate × up
```

The multiplication is element-wise.

Finally:

```text
output = W_down hidden
```

Tensor shapes:

```text
Input:
[B,T,C]

Gate:
[B,T,F]

Up:
[B,T,F]

Hidden:
[B,T,F]

Output:
[B,T,C]
```

For current dimensions:

```text
128 → 384 → 128
```

The MLP does not directly mix different token positions.

It transforms features independently at each position using shared weights.

Attention performs token-to-token communication.

The MLP performs nonlinear feature computation within each token.

## 14.6 `block.py`

This file defines one complete `TransformerBlock`.

The architecture is Pre-Norm:

```text
x
│
├──────── residual ──────────────┐
│                                │
▼                                │
RMSNorm                          │
▼                                │
Multi-head causal attention      │
▼                                │
Add ◀────────────────────────────┘
│
▼
x'
│
├──────── residual ──────────────┐
│                                │
▼                                │
RMSNorm                          │
▼                                │
SwiGLU                           │
▼                                │
Add ◀────────────────────────────┘
│
▼
output
```

The forward method conceptually performs:

```python
x = x + attention(attention_norm(x))
x = x + mlp(mlp_norm(x))
return x
```

The residual stream always retains the original representation and adds the sublayer’s contribution.

Input and output remain:

```text
[B,T,C]
```

## 14.7 `transformer.py`

This file defines `TransformerStack`.

It creates:

```python
nn.ModuleList([
    TransformerBlock(config)
    for _ in range(config.n_layers)
])
```

`ModuleList` ensures all block parameters are:

- Registered
- Included in `model.parameters()`
- Moved by `model.to(device)`
- Saved in `state_dict()`

For the current model, it creates four independent blocks.

They share architecture but not weights.

The forward pass loops through them:

```text
x0
↓ block 0
x1
↓ block 1
x2
↓ block 2
x3
↓ block 3
x4
```

Shape stays `[B,T,C]` throughout.

## 14.8 `gpt.py`

This file defines the complete `TinyGPT` neural network.

### Components

```text
TokenEmbedding
TransformerStack
Final RMSNorm
LM head
```

### Forward path

Input:

```text
token_ids: [B,T]
dtype: torch.long
```

Validation checks:

- Exactly two dimensions
- Correct integer dtype
- Sequence does not exceed context length
- Token IDs are nonnegative
- Token IDs are less than vocabulary size

Then:

```text
Token IDs
[B,T]
↓
TokenEmbedding
[B,T,C]
↓
TransformerStack
[B,T,C]
↓
Final RMSNorm
[B,T,C]
↓
LM Head
[B,T,V]
```

### LM head

The language-model head is:

```text
Linear(C → V)
```

It produces one logit per vocabulary token at every sequence position.

### Weight tying

The LM-head weight is shared with the token-embedding weight.

Both have shape:

```text
[V,C]
```

The embedding uses this matrix as a row-lookup table.

The LM head uses it as an output projection.

The same `Parameter` object is referenced in both places.

This:

- Reduces parameter count
- Couples input and output token representations
- Causes gradients from both roles to accumulate into the same shared matrix

### Initialization

Linear and embedding weights are initialized from a normal distribution:

```text
mean = 0
standard deviation = 0.02
```

The clean order is:

1. Construct modules
2. Initialize modules
3. Tie LM-head weight to embedding weight

RMSNorm weights remain initialized to ones.

### Causality

A complete-model causality check uses two sequences with the same prefix and different future tokens.

Logits for prefix positions must remain identical.

This proves that future tokens cannot influence earlier predictions through any of the four blocks.

---

# 15. Parameter count

Let:

```text
V = vocabulary size
C = d_model = 128
F = d_ff = 384
L = number of blocks = 4
```

## 15.1 Token embedding

```text
V × C
```

For `V=1024`:

```text
1024 × 128 = 131,072
```

## 15.2 Attention per block

Fused QKV:

```text
C × 3C
= 128 × 384
= 49,152
```

Output projection:

```text
C × C
= 128 × 128
= 16,384
```

Attention total:

```text
65,536
```

## 15.3 SwiGLU per block

Gate:

```text
C × F
= 128 × 384
= 49,152
```

Up:

```text
C × F
= 49,152
```

Down:

```text
F × C
= 49,152
```

SwiGLU total:

```text
147,456
```

## 15.4 RMSNorm per block

Two RMSNorm layers:

```text
2 × C
= 256
```

## 15.5 One Transformer block

```text
65,536
+ 147,456
+ 256
= 213,248
```

## 15.6 Four blocks

```text
4 × 213,248
= 852,992
```

## 15.7 Final norm

```text
128
```

## 15.8 Total with weight tying

For vocabulary size 1024:

```text
Token embedding:        131,072
Transformer stack:      852,992
Final RMSNorm:              128
LM head additional:           0
--------------------------------
Total:                  984,192
```

The LM head adds no unique matrix because it shares the embedding parameter.

The actual total changes with actual tokenizer vocabulary size.

---

# 16. Training package: `tinygpt/training/`

## 16.1 `loss.py`

This file defines `language_model_loss(logits, targets)`.

Expected shapes:

```text
logits:  [B,T,V]
targets: [B,T]
```

Validation checks:

- Logits have three dimensions
- Targets have two dimensions
- Batch and sequence dimensions match
- Targets use `torch.long`

Cross-entropy expects examples in the form:

```text
[N,V]
```

and target IDs in:

```text
[N]
```

Therefore:

```text
[B,T,V]
→ [B × T,V]
```

and:

```text
[B,T]
→ [B × T]
```

The loss calls:

```python
F.cross_entropy(...)
```

with raw logits.

It does not apply softmax manually because cross-entropy already performs the numerically stable log-softmax and negative-log-likelihood computation internally.

For one prediction:

\[
loss=-\log P(\text{correct token})
\]

If the model assigns high probability to the correct token, loss is small.

If it assigns very low probability to the correct token, loss is large.

A roughly uniform untrained model has expected loss near:

\[
\log(V)
\]

For `V=1024`:

```text
approximately 6.93
```

## 16.2 `optimizer.py`

This file creates AdamW with two parameter groups.

### Decay group

Parameters with:

```text
ndim >= 2
```

receive weight decay.

These generally include:

- Embedding matrix
- QKV matrix
- Attention output projection
- SwiGLU matrices

### No-decay group

One-dimensional parameters do not receive weight decay.

These are primarily RMSNorm scaling vectors.

### Why AdamW

AdamW tracks running estimates of:

- Gradient direction
- Squared-gradient magnitude
- Per-parameter optimization step

It uses adaptive updates and decoupled weight decay.

The optimizer itself has state that must be checkpointed.

### Tied-weight handling

Because PyTorch’s parameter traversal deduplicates shared parameters, the tied embedding/LM-head matrix should enter the optimizer only once.

A diagnostic check confirms every trainable parameter is covered exactly once.

## 16.3 `step.py`

This file defines one complete `train_step()`.

Sequence:

```text
model.train()
↓
optimizer.zero_grad(set_to_none=True)
↓
logits = model(x)
↓
loss = language_model_loss(logits, y)
↓
loss.backward()
↓
clip_grad_norm_
↓
optimizer.step()
↓
return scalar metrics
```

### `zero_grad(set_to_none=True)`

Gradients accumulate by default.

Setting them to `None` prevents accidental accumulation between ordinary training steps and avoids unnecessary zero-fill work.

### `loss.backward()`

Calculates gradients through:

```text
Cross-entropy
LM head
Final RMSNorm
Block 4
Block 3
Block 2
Block 1
Token embedding
```

### Gradient clipping

`clip_grad_norm_()` calculates a global gradient norm.

If it exceeds the configured maximum, all gradients are scaled down while preserving their relative direction.

The function returns the original norm, which is logged.

### `optimizer.step()`

This is the operation that actually changes model parameters.

`backward()` calculates how weights should change.

`optimizer.step()` performs the change.

## 16.4 `schedule.py`

This file controls learning rate by training step.

### Warmup

For the first configured number of steps, the learning rate rises linearly.

With 50 warmup steps:

```text
Step 1  → small LR
...
Step 50 → maximum LR
```

### Cosine decay

After warmup, learning rate gradually decreases from:

```text
0.0003
```

toward:

```text
0.00003
```

using a cosine curve.

### `get_learning_rate(step, config)`

Returns the learning rate for a specific zero-based training step.

### `set_learning_rate(optimizer, learning_rate)`

Updates all optimizer parameter groups while preserving their separate weight-decay values.

## 16.5 `evaluate.py`

This file defines `evaluate_loss()`.

It:

1. Records whether the model was already in training mode
2. Calls `model.eval()`
3. Creates a fixed evaluation generator from a specified seed
4. Enters `torch.no_grad()`
5. Samples several evaluation batches
6. Calculates loss for each
7. Averages the losses
8. Calculates perplexity
9. Restores training mode if necessary

Perplexity is:

\[
e^{loss}
\]

The same fixed evaluation seed is reused at different checkpoints.

This means evaluation compares different model states on the same sampled windows.

Validation sampling therefore does not disturb the training batch sequence.

## 16.6 `checkpoint.py`

This file manages persistent training state.

### Captured state

A checkpoint includes:

```text
checkpoint format version
completed optimizer steps
best validation loss
model configuration
training configuration
tokenizer SHA-256
token-data metadata SHA-256
model state_dict
optimizer state_dict
random-number-generator state
latest evaluation metrics
```

### RNG state

It captures:

```text
Python random state
PyTorch global RNG state
CUDA RNG states, when available
Training generator state
```

The training generator state is essential because random batch positions are sampled through an independent `torch.Generator`.

### Atomic save

The checkpoint is first written to a temporary file:

```text
latest.pt.tmp
```

After a successful write, `os.replace()` replaces the real checkpoint.

This reduces the chance that an interrupted write destroys the last valid checkpoint.

### Validation before load

Resume checks:

- Checkpoint version
- Exact model configuration
- Exact training configuration
- Tokenizer fingerprint
- Token-data metadata fingerprint

Resume is intentionally strict.

It means:

```text
Continue exactly the same experiment
```

not:

```text
Silently change architecture or optimizer settings
```

### Loading

The loader:

1. Reads the checkpoint onto CPU
2. Validates compatibility
3. Loads model weights
4. Loads AdamW state
5. Moves model to the runtime device
6. Moves optimizer-state tensors to the same device
7. Restores RNG state
8. Returns checkpoint metadata

### `write_run_metadata()`

Creates a human-readable JSON file with:

```text
experiment name
model configuration
training configuration
tokenizer hash
token-data hash
Python version
PyTorch version
```

---

# 17. Training orchestrator: `scripts/train.py`

This is the project’s composition root for pretraining.

It connects otherwise independent modules.

The uploaded file confirms that the entrypoint imports:

- Model and training configuration
- `TokenDataset`
- `TinyGPT`
- `BPETokenizer`
- Evaluation
- Optimizer construction
- Learning-rate scheduling
- Training-step logic
- Device selection
- Parameter reporting
- Random seeding
- Checkpoint management
- File hashing fileciteturn0file0L5-L39 fileciteturn0file0L44-L57

It also loads the tokenizer, calculates fingerprints, creates model configuration from tokenizer vocabulary size, loads train and validation token datasets, constructs the model, constructs AdamW, and creates a dedicated training generator. fileciteturn0file0L115-L188

The corrected lifecycle should be understood as follows.

## 17.1 Startup

1. Construct `TrainingConfig`
2. Construct `RunConfig`
3. Seed Python and PyTorch
4. Select device
5. Load tokenizer
6. Calculate tokenizer and token-metadata hashes
7. Construct `ModelConfig` using actual tokenizer vocabulary size
8. Load train and validation `TokenDataset`
9. Verify both splits contain at least `context_length + 1` tokens
10. Construct `TinyGPT`
11. Move model to device
12. Construct AdamW
13. Construct deterministic training generator
14. Construct run directory paths

## 17.2 Fresh-run safety

If:

```text
resume_from = None
```

but the experiment directory already contains:

```text
run_metadata.json
latest.pt
best.pt
```

the script raises:

```text
FileExistsError
```

This is intentional.

It prevents a fresh run from overwriting an existing experiment.

The developer must make an explicit decision:

```text
Resume existing run
or
Use a new experiment name
or
Delete stale artifacts for a deliberate clean restart
```

This safety mechanism was encountered and resolved during the project.

## 17.3 Fresh run

A fresh run:

1. Sets `start_step = 0`
2. Sets `best_val_loss = infinity`
3. Writes `run_metadata.json`
4. Evaluates untrained model on fixed train and validation batches
5. Sets the initial validation loss as the first best loss
6. Saves step-zero `latest.pt`
7. Saves step-zero `best.pt`
8. Starts the optimization loop

## 17.4 Resume run

A resume run:

1. Loads `latest.pt`
2. Restores model state
3. Restores AdamW state
4. Restores RNG state
5. Restores training-generator state
6. Restores completed step
7. Restores best validation loss
8. Restores latest evaluation metadata
9. Continues from the next correct scheduler position

If 350 updates are complete:

```text
start_step = 350
```

The next loop iteration performs optimizer update 351.

Warmup does not restart.

## 17.5 Training loop

For each step:

1. Calculate learning rate
2. Set it on all optimizer groups
3. Sample a training batch
4. Perform `train_step()`
5. Add processed tokens to throughput counter
6. Log metrics if due
7. Evaluate if due
8. Update best validation loss
9. Save `best.pt` if validation improved
10. Save `latest.pt` if checkpointing is due

Correct ordering matters:

```text
Training update
↓
Optional evaluation
↓
Update best bookkeeping
↓
Optional best checkpoint
↓
Optional latest checkpoint
```

This ensures `latest.pt` contains current `best_val_loss` and current `last_eval`.

## 17.6 Logged metrics

Training logs include:

```text
completed step
current batch loss
learning rate
gradient norm
tokens per second
```

Periodic evaluation logs include:

```text
average train evaluation loss
train perplexity
average validation loss
validation perplexity
```

The current batch loss is noisy.

The fixed multi-batch evaluation losses are more useful for trend analysis.

---

# 18. Complete training objective

## 18.1 Input and target relationship

Given a token stream:

```text
[10,20,30,40,50]
```

a context-length-four example is:

```text
Input:
[10,20,30,40]

Target:
[20,30,40,50]
```

The model produces a distribution at every input position.

Because attention is causal:

```text
Position 0 sees token 10
and predicts token 20

Position 1 sees tokens 10,20
and predicts token 30

Position 2 sees tokens 10,20,30
and predicts token 40

Position 3 sees tokens 10,20,30,40
and predicts token 50
```

One sequence therefore supplies multiple next-token learning targets.

## 18.2 One optimizer step with default dimensions

Batch:

```text
x: [8,128]
y: [8,128]
```

Forward output:

```text
logits: [8,128,V]
```

Flattened:

```text
logits: [1024,V]
targets: [1024]
```

One optimizer step therefore contains 1024 next-token classification examples.

## 18.3 Forward pass

```text
x
[B,T]
↓
Embedding
[B,T,C]
↓
4 Transformer blocks
[B,T,C]
↓
Final RMSNorm
[B,T,C]
↓
LM head
[B,T,V]
↓
Cross-entropy against y
scalar loss
```

## 18.4 Backward pass

The scalar loss sends gradients backward through the entire graph:

```text
Loss
↓
LM head / shared embedding matrix
↓
Final RMSNorm
↓
Block 4 MLP and attention
↓
Block 3 MLP and attention
↓
Block 2 MLP and attention
↓
Block 1 MLP and attention
↓
Token embedding
```

RoPE has no trainable parameters, but gradients pass through its rotations into Q and K projection weights.

The causal mask has no trainable parameters.

## 18.5 Parameter update

AdamW receives:

```text
current parameter
current gradient
first-moment history
second-moment history
learning rate
weight-decay setting
```

It updates every trainable parameter.

One step produces only small changes.

Language ability emerges from many updates over many token examples.

---

# 19. How to interpret training behavior

## 19.1 Healthy learning

```text
Train loss decreases
Validation loss decreases
```

This suggests the model is learning patterns that generalize beyond sampled training windows.

## 19.2 Overfitting

```text
Train loss keeps decreasing
Validation loss begins increasing
```

This suggests memorization of the training subset.

`best.pt` preserves the lowest observed validation loss even if later training overfits.

## 19.3 Flat loss

Possible causes:

- Learning rate too low
- Dataset too small or poorly prepared
- Targets not shifted correctly
- Gradients not reaching parameters
- Optimizer missing parameters
- Parameters not changing
- Severe architecture bug
- Inadequate number of training steps

## 19.4 NaN or infinity

Possible causes:

- Numerical instability
- Excessive learning rate
- Exploding gradients
- Invalid logits
- Invalid input token IDs
- Division or masking error

Gradient clipping and normalization reduce risk but do not make every failure impossible.

## 19.5 Overfit-one-batch diagnostic

Repeatedly training on one tiny fixed batch is a deliberate debugging test.

A functioning model should be able to drive that batch’s loss downward.

If it cannot, the system may have a fundamental problem in:

- Data-target alignment
- Forward pass
- Loss
- Gradient flow
- Optimizer setup
- Learning rate

---

# 20. Checkpoint artifacts

## 20.1 `run_metadata.json`

Human-readable experiment identity.

Contains configuration and dependency information.

It is useful for inspection without loading PyTorch.

## 20.2 `latest.pt`

Represents a recent recoverable training state.

Use it for:

```text
Resume training
```

It may not be the best generalizing model.

## 20.3 `best.pt`

Represents the evaluated checkpoint with the lowest validation loss.

Use it for:

```text
Generation
Final validation inspection
Base-model evaluation
```

Example:

```text
Step 500 validation loss: 3.8
Step 1000 validation loss: 4.4
```

Then:

```text
best.pt   → step 500
latest.pt → step 1000
```

## 20.4 Why optimizer state matters

If only model weights are restored, AdamW loses its running moments.

Training would continue from the same model but not the same optimization process.

A true resume restores both.

## 20.5 Why RNG state matters

Without RNG restoration, the next training batches after resume differ from those that would have appeared in uninterrupted training.

Saving the dedicated training generator allows the batch sequence to continue reproducibly.

---

# 21. Generation package: `tinygpt/generation/`

## 21.1 `load.py`

This file reconstructs a trained model for inference.

`load_model_for_generation()`:

1. Verifies checkpoint existence
2. Loads tokenizer
3. Loads checkpoint onto CPU
4. Verifies required checkpoint fields
5. Calculates current tokenizer hash
6. Compares it with checkpoint tokenizer hash
7. Reconstructs `ModelConfig` from checkpoint values
8. Verifies tokenizer vocabulary matches model vocabulary
9. Constructs `TinyGPT`
10. Loads model state
11. Moves model to the requested device
12. Sets evaluation mode
13. Returns model, tokenizer, and checkpoint metadata

Generation does not need to restore AdamW.

## 21.2 `sampling.py`

This file contains decoding policies.

### `apply_top_k()`

Keeps only the `k` highest logits.

All others become:

```text
-inf
```

After softmax, excluded tokens receive probability zero.

### `apply_top_p()`

Sorts tokens by descending probability.

It keeps the smallest high-probability set whose cumulative probability reaches the configured threshold.

For:

```text
top_p = 0.95
```

approximately 95% of probability mass remains available.

The implementation ensures at least the highest-probability token remains.

### `sample_next_token()`

Inputs:

```text
logits
temperature
top_k
top_p
optional random generator
```

Behavior:

- `temperature == 0` uses greedy `argmax`
- Otherwise divide logits by temperature
- Apply top-k
- Apply top-p
- Apply softmax
- Sample one token using `torch.multinomial`

### Temperature

Lower than 1:

```text
Sharper distribution
More deterministic
```

Greater than 1:

```text
Flatter distribution
More random
```

Temperature changes decoding behavior.

It does not change model weights or knowledge.

## 21.3 `generate.py`

This file contains the autoregressive generation loop.

### Input

```text
trained model
tokenizer
prompt string
maximum new tokens
device
temperature
top-k
top-p
optional random generator
```

### Prompt encoding

The prompt becomes token IDs without EOS:

```text
"Once upon a time"
↓
[... token IDs ...]
```

A batch dimension is added:

```text
[T]
→ [1,T]
```

### Repeated generation loop

For each new token:

1. Keep only the last `context_length` token IDs
2. Run the model
3. Obtain logits with shape `[1,T,V]`
4. Select only the last position:

   ```text
   [1,V]
   ```

5. Sample or greedily choose one token
6. Append the new token:

   ```text
   [1,T] + [1,1] → [1,T+1]
   ```

7. Stop if EOS was generated
8. Otherwise repeat

### Why the last position is used

During generation, all prompt tokens are already known.

Only the distribution after the final current token is needed.

### Context truncation

If the generated sequence becomes longer than 128 tokens, the model receives only the latest 128.

The complete token sequence is still retained for final decoding.

### Decoding

At the end:

- Full token sequence is decoded
- Newly generated token portion is decoded
- Token IDs are returned
- New-token count is returned
- EOS-stop status is returned

The full sequence is decoded together because byte-level BPE tokens may not each be valid standalone UTF-8 strings.

---

# 22. Complete generation process

Suppose the prompt is:

```text
Once upon a time
```

## Step 1

Tokenizer:

```text
Prompt
↓
[312, 487, 91, 623]
```

Tensor:

```text
[1,4]
```

## Step 2

Model:

```text
[1,4]
↓
TinyGPT
↓
[1,4,V]
```

## Step 3

Take last position:

```text
logits[:, -1, :]
→ [1,V]
```

## Step 4

Apply decoding policy:

```text
temperature
top-k
top-p
softmax
multinomial
```

Suppose token 710 is selected.

## Step 5

Append:

```text
[312,487,91,623]
+
[710]
```

## Step 6

Run again:

```text
[1,5]
↓
TinyGPT
↓
[1,5,V]
```

Take final-position logits again.

This is autoregression:

> The model’s previous output becomes part of its next input.

---

# 23. Current generation limitations

## 23.1 No KV cache

Every new token recomputes the whole visible context.

Generating token 100 may require processing all visible preceding tokens again.

Production systems cache previous Keys and Values.

The current implementation is intentionally simple and transparent.

## 23.2 No streaming decoder

Output is decoded after generation completes.

A production interface would stream pieces while safely managing partial UTF-8 byte sequences.

## 23.3 No repetition controls

There is currently no:

- Repetition penalty
- Frequency penalty
- Presence penalty
- No-repeat n-gram rule

Small models may repeat phrases.

## 23.4 Context positions restart after truncation

When generation exceeds the context window, only the latest 128 tokens are passed, and their RoPE positions begin again from zero.

This is acceptable for the current local-context learning implementation.

A KV-cached implementation should explicitly manage positional offsets.

## 23.5 Base model only

The model completes text.

It has not been trained to interpret a prompt as an instruction from a user.

---

# 24. Script catalog

The `scripts/` directory contains entrypoints and focused verification programs.

These scripts are intentionally small so failures can be isolated.

## 24.1 Environment and tensor foundations

| File | Internal behavior and purpose |
|---|---|
| `check_environment.py` | Prints Python version, OS information, and exact interpreter path. Confirms VS Code and terminal use `.venv`. |
| `check_torch.py` | Imports PyTorch, prints version and selected device, creates a test tensor, moves it to the device, and prints its shape/device. |
| `tensor_basics.py` | Demonstrates scalars, vectors, matrices, three-dimensional tensors, `[B,T,C]`, indexing, slicing, reshape, transpose, matrix multiplication, broadcasting, stacking, concatenation, and softmax. |
| `autograd_basics.py` | Creates a scalar with `requires_grad=True`, calculates `x²`, calls `backward()`, and verifies the derivative at `x=3` is 6. |

## 24.2 Configuration and reproducibility

| File | Internal behavior and purpose |
|---|---|
| `check_config.py` | Constructs model and training configurations, prints them, checks calculated `head_dim`, and can deliberately test invalid divisibility. |
| `check_seed.py` | Resets the seed twice and proves the same random tensor is generated both times. |
| `check_experiment.py` | Combines configuration, seeding, device detection, and a reproducible random tensor into one experiment bootstrap check. |

## 24.3 Dataset acquisition and preparation

| File | Internal behavior and purpose |
|---|---|
| `download_tinystories.py` | Streams TinyStories examples, writes stories separated by blank lines, tracks UTF-8 bytes, and stops near the configured target size, approximately 5 MiB. |
| `prepare_data.py` | Loads raw text, normalizes line endings, splits 90/5/5, verifies lengths, writes train/val/test files, and creates processed metadata. |
| `check_dataset_sizes.py` | Loads train, validation, and test token tensors and prints token counts. Confirms validation is comfortably larger than the context window. |

## 24.4 Tokenization learning scripts

| File | Internal behavior and purpose |
|---|---|
| `check_char_tokenizer.py` | Builds a character vocabulary from training text, prints visible representations of early vocabulary entries, and verifies encode/decode round trip. |
| `check_bytes.py` | Displays UTF-8 byte sequences for ASCII, accented text, and emoji. Demonstrates that one Unicode character can require multiple bytes. |
| `check_byte_tokenizer.py` | Encodes and decodes multilingual and emoji examples using raw byte IDs and confirms no text OOV problem. |
| `inspect_tokenization.py` | Encodes a training-text sample with the byte tokenizer, compares character and token counts, prints IDs, and verifies round trip. |
| `check_bpe.py` | Trains a very small BPE tokenizer on repetitive text such as `banana bandana`, compares raw bytes with BPE-token count, and verifies exact decode. |
| `train_tokenizer.py` | Loads representative training text, takes a bounded tokenizer-training subset, trains byte-level BPE, calculates compression statistics, verifies round trip, and saves `tokenizer.json`. |
| `check_saved_tokenizer.py` | Reloads `tokenizer.json` and tests English, accented text, Hindi, and emoji. Confirms persistence and byte fallback. |

## 24.5 Token dataset and batching scripts

| File | Internal behavior and purpose |
|---|---|
| `prepare_tokens.py` | Loads the frozen tokenizer, reads each split, separates documents by blank lines, encodes each document with EOS, saves one-dimensional `torch.long` tensors, and writes token metadata. |
| `check_token_dataset.py` | Loads one token split, retrieves one window, prints input/target IDs, decodes them, and demonstrates the one-token shift. |
| `check_batch.py` | Samples a reproducible mini-batch, prints `[B,T]` shapes and dtypes, and decodes the first input/target pair. |

## 24.6 Embedding and positional scripts

| File | Internal behavior and purpose |
|---|---|
| `check_embeddings.py` | Constructs token embeddings, prints matrix shape, encodes sample text, verifies `[B,T] → [B,T,C]`, and proves forward output equals direct row lookup. |
| `check_embedding_gradients.py` | Uses repeated token IDs and a simple fake loss to show that only used embedding rows receive gradients and repeated IDs accumulate larger gradients. |
| `check_model_input.py` | Connects real dataset batch to the embedding layer and verifies `[8,128] → [8,128,128]`. |
| `check_positions.py` | Shows that repeated token IDs have identical raw embeddings but different representations after adding learned positional embeddings. |
| `check_rope_math.py` | Prints RoPE dimension indices, inverse frequencies, position-angle matrix, cosine values, and sine values. |
| `check_rope.py` | Applies RoPE to `[B,H,T,D]`, verifies shape preservation, position-zero identity, later-position change, and norm preservation. |

## 24.7 Attention scripts

| File | Internal behavior and purpose |
|---|---|
| `check_qkv.py` | Creates separate Query, Key, and Value projections; prints weight and output shapes; calculates raw `QKᵀ` attention scores. |
| `check_attention_head.py` | Runs one causal attention head, prints output shape and parameter count, inspects attention weights, verifies rows sum to one, and confirms future attention is zero. |
| `check_multihead_attention.py` | Runs fused multi-head attention, verifies `[B,T,C] → [B,H,T,T] → [B,T,C]`, checks parameter count, causal masking, and row normalization. |

## 24.8 Normalization and MLP scripts

| File | Internal behavior and purpose |
|---|---|
| `check_rmsnorm.py` | Applies RMSNorm to scaled random values, verifies shape preservation, parameter count, output RMS near one at initialization, and norm-weight gradients. |
| `check_attention_residual.py` | Applies Pre-Norm attention and adds the residual, confirming all shapes remain `[B,T,C]`. |
| `check_silu.py` | Prints SiLU output for negative, zero, and positive values. Demonstrates smooth nonlinear behavior. |
| `check_swiglu.py` | Runs the SwiGLU network, prints hidden and output shapes, and verifies its parameter count. |
| `check_mlp_independence.py` | Changes one token position and proves the MLP changes that position but not other positions, demonstrating position-wise processing. |
| `check_mlp_residual.py` | Applies RMSNorm, SwiGLU, and residual addition and verifies shape preservation. |

## 24.9 Transformer and complete-model scripts

| File | Internal behavior and purpose |
|---|---|
| `check_transformer_block.py` | Runs one complete Transformer block, verifies input/output shape equality, and confirms expected parameter count. |
| `check_block_gradients.py` | Uses a fake scalar loss to prove gradients reach block input, QKV weights, MLP gate weights, and norm weights. |
| `check_transformer_stack.py` | Creates four blocks, verifies independent weights, stack output shape, and total stack parameter count. |
| `check_stack_gradients.py` | Proves gradients reach QKV matrices in every block. |
| `check_tinygpt.py` | Constructs the complete model, passes random token IDs, prints `[B,T,V]` logits, parameter counts, and parameter-memory estimate. |
| `check_weight_tying.py` | Verifies embedding weight and LM-head weight are the same Python object and share the same memory pointer. |
| `check_full_forward.py` | Loads a real token batch, runs the complete model, and prints input, target, logit shapes, devices, and dtypes. |
| `check_model_causality.py` | Uses two sequences with identical prefixes and different futures, proving prefix logits remain identical. |

## 24.10 Loss and gradient scripts

| File | Internal behavior and purpose |
|---|---|
| `check_cross_entropy.py` | Demonstrates how correct-token confidence changes cross-entropy loss. |
| `check_tinygpt_loss.py` | Calculates real language-model loss on a dataset batch and compares it with `log(vocab_size)`. |
| `check_language_gradients.py` | Calls backward on real next-token loss and confirms gradients exist in embedding, early attention, late MLP, and final norm. |
| `check_gradient_descent.py` | Manually updates one scalar parameter using ordinary gradient descent to explain the optimizer concept. |

## 24.11 Optimizer and learning scripts

| File | Internal behavior and purpose |
|---|---|
| `check_optimizer_groups.py` | Prints decay and no-decay parameter groups and verifies every trainable parameter appears exactly once. |
| `check_optimizer_step.py` | Saves a QKV weight snapshot, runs one real training step, compares before/after values, and proves parameters changed. |
| `check_overfit_batch.py` | Repeats optimization on one fixed batch and prints loss over time to prove the model can learn and memorize a tiny example. |
| `check_lr_schedule.py` | Prints learning rate at selected warmup, peak, decay, midpoint, and final steps. |

## 24.12 Checkpoint and training scripts

| File | Internal behavior and purpose |
|---|---|
| `check_checkpoint_roundtrip.py` | Trains one step, stores reference logits and RNG state, saves a checkpoint, creates fresh model/optimizer/generator objects, restores them, and verifies identical logits and restored generator state. |
| `train.py` | Runs full pretraining, fixed evaluation, logging, throughput measurement, learning-rate scheduling, best-checkpoint selection, latest-checkpoint saving, and resume. |

## 24.13 Generation scripts

| File | Internal behavior and purpose |
|---|---|
| `generate_greedy.py` | Loads `best.pt`, uses deterministic argmax decoding, and prints one continuation. |
| `generate.py` | Command-line interface accepting prompt, token limit, temperature, top-k, top-p, seed, and checkpoint path. |
| `check_generation.py` | Creates two generators with the same seed and proves identical sampled token sequences are produced. |

---

# 25. End-to-end execution runbook

## 25.1 Activate environment

```powershell
.venv\Scripts\Activate.ps1
```

## 25.2 Download the TinyStories subset

```powershell
python -m scripts.download_tinystories
```

Produces:

```text
data/raw/input.txt
```

## 25.3 Normalize and split

```powershell
python -m scripts.prepare_data
```

Produces:

```text
data/processed/train.txt
data/processed/val.txt
data/processed/test.txt
data/processed/metadata.json
```

## 25.4 Train the tokenizer

```powershell
python -m scripts.train_tokenizer
```

Produces:

```text
data/tokenizer/tokenizer.json
```

## 25.5 Encode the corpus

```powershell
python -m scripts.prepare_tokens
```

Produces:

```text
data/tokens/train.pt
data/tokens/val.pt
data/tokens/test.pt
data/tokens/metadata.json
```

## 25.6 Check split sizes

```powershell
python -m scripts.check_dataset_sizes
```

Validation must contain substantially more than:

```text
129 tokens
```

for a context length of 128.

## 25.7 Verify checkpoint behavior

```powershell
python -m scripts.check_checkpoint_roundtrip
```

Expected critical results:

```text
Restored step: 1
Logits exactly identical: True
Training generator restored: True
```

## 25.8 Start fresh training

Ensure:

```text
RunConfig.resume_from = None
```

and the chosen experiment directory does not already contain artifacts.

Then:

```powershell
python -m scripts.train
```

## 25.9 Resume training

Set:

```text
resume_from =
checkpoints/tinystories_5mb_v1/latest.pt
```

Then run:

```powershell
python -m scripts.train
```

## 25.10 Generate text

Greedy:

```powershell
python -m scripts.generate --prompt "Once upon a time" --max-new-tokens 100 --temperature 0
```

Sampling:

```powershell
python -m scripts.generate --prompt "Once upon a time" --max-new-tokens 100 --temperature 0.8 --top-k 40 --top-p 0.95
```

---

# 26. Complete data-to-text trace

This section follows one piece of information through the entire system.

## 26.1 Raw story

```text
Once upon a time there was a little girl.
```

## 26.2 UTF-8 representation

The string becomes bytes.

ASCII characters use one byte each.

Other Unicode characters may use multiple bytes.

## 26.3 BPE encoding

The tokenizer starts with the bytes.

It applies learned merge rules in rank order.

Output might conceptually be:

```text
[401, 288, 91, 612, 310, ...]
```

## 26.4 Stored token stream

The story token IDs are appended to the split’s one-dimensional tensor.

EOS is appended:

```text
[story tokens..., eos_token_id]
```

## 26.5 Training window

A random context is sampled:

```text
x = [401, 288, 91, 612]
y = [288, 91, 612, 310]
```

Shapes after batching:

```text
x: [B,T]
y: [B,T]
```

## 26.6 Embedding

Each token ID selects one row of the shared embedding matrix.

```text
[B,T]
→ [B,T,C]
```

## 26.7 Transformer Block 1

### Normalize

Each token vector is RMS-normalized.

### QKV projection

```text
[B,T,C]
→ [B,T,3C]
```

### Split and reshape

```text
Q,K,V:
[B,T,C]
→ [B,H,T,D]
```

### RoPE

Q and K are rotated by position.

### Scores

```text
QKᵀ:
[B,H,T,D] @ [B,H,D,T]
→ [B,H,T,T]
```

### Scale, mask, softmax

Future scores become zero probability.

### Aggregate Values

```text
[B,H,T,T] @ [B,H,T,D]
→ [B,H,T,D]
```

### Merge heads

```text
[B,H,T,D]
→ [B,T,C]
```

### Output projection and residual

Attention contribution is added to the residual stream.

### MLP sublayer

Normalized token features expand to `F`, are gated with SwiGLU, reduce back to `C`, and are added to the residual stream.

## 26.8 Blocks 2–4

Each repeats the same architecture with independent weights.

## 26.9 Final norm

Final hidden representations are normalized.

## 26.10 LM head

Each token position’s `C` features are projected to `V` logits.

```text
[B,T,C]
→ [B,T,V]
```

## 26.11 Loss

At each position, cross-entropy selects the logit corresponding to the correct next-token ID.

The mean loss across `B × T` positions is returned.

## 26.12 Backpropagation and update

Gradients are calculated and clipped.

AdamW changes parameters.

After many updates, the model learns patterns from the corpus.

## 26.13 Generation

A prompt is encoded and sent through the same model.

Only the final position’s logits are used.

A next token is selected, appended, and fed back into the model.

The final token list is decoded into text.

---

# 27. Important invariants

These conditions should always be true.

## Configuration

```text
d_model % n_heads == 0
head_dim % 2 == 0
warmup_steps < max_steps
minimum learning rate <= maximum learning rate
split fractions sum to 1
```

## Tokenizer

```text
decode(encode(text)) == text
all base byte IDs are representable
tokenizer vocabulary matches model vocabulary
tokenizer hash matches checkpoint hash
```

## Dataset

```text
Stored token tensor is one-dimensional
Stored token dtype is torch.long
Each split contains at least context_length + 1 tokens
Input and target shapes match
Targets are inputs shifted by one token
```

## Model

```text
Input token IDs have shape [B,T]
T <= context_length
All token IDs are in [0,V)
Embedding output has shape [B,T,C]
Attention output has shape [B,T,C]
Transformer block preserves [B,T,C]
Final logits have shape [B,T,V]
```

## Attention

```text
Every attention row sums approximately to 1
Future attention probabilities are zero
Q and K receive RoPE
V does not receive RoPE
```

## Training

```text
Loss is finite
Gradients exist
Every optimizer parameter appears exactly once
optimizer.step changes parameter values
Validation does not call backward
```

## Checkpointing

```text
Checkpoint model config matches current model config
Training config matches
Tokenizer hash matches
Token metadata hash matches
Resume starts from completed_step
```

## Generation

```text
Checkpoint tokenizer matches active tokenizer
Only last-position logits select next token
Context never exceeds configured maximum
EOS can terminate generation
Same seed and settings reproduce the same CPU sample
```

---

# 28. Common failure modes

## 28.1 Validation dataset too small

Cause:

```text
len(val_tokens) < context_length + 1
```

Correct fix:

- Increase corpus size
- Prepare data again
- Retrain tokenizer if corpus changed
- Re-encode all splits

Temporary alternative:

- Reduce context length

## 28.2 Existing experiment-directory error

Cause:

```text
resume_from is None
```

while run artifacts already exist.

Correct choices:

- Set `resume_from` to `latest.pt`
- Choose a new experiment name
- Deliberately delete stale artifacts for a clean restart

Do not remove the safety guard.

## 28.3 Token ID exceeds vocabulary

Likely causes:

- Token data was produced by a different tokenizer
- Model was constructed with the wrong vocabulary size
- Old `.pt` files remained after tokenizer retraining

Correct fix:

Regenerate token files using the active tokenizer.

## 28.4 Tokenizer mismatch on checkpoint load

Cause:

```text
current tokenizer SHA-256
!=
checkpoint tokenizer SHA-256
```

Correct fix:

Use the exact tokenizer paired with the checkpoint.

## 28.5 Loss stays near random baseline

Possible causes:

- Insufficient steps
- Learning rate issue
- Dataset too limited
- Optimizer not updating
- Target shift error
- Gradient-flow problem

Run the focused diagnostic scripts.

## 28.6 Loss becomes NaN

Inspect:

- Learning rate
- Gradient norm
- Input token range
- Masking
- Intermediate values
- Parameter initialization

## 28.7 Generation is repetitive

Possible causes:

- Model is tiny
- Dataset is small
- Training is insufficient
- Model overfit
- Greedy decoding
- Temperature too low

Sampling controls can help diversity but cannot create knowledge that the model never learned.

## 28.8 Generation is random nonsense

Possible causes:

- Using step-zero checkpoint
- Too little training
- Wrong tokenizer
- Loading wrong checkpoint
- Validation loss remained close to random baseline

---

# 29. Scalability design already present

Although training currently happens on one CPU, several architectural decisions support future scaling.

## Device abstraction

The model does not hardcode CPU.

## Configuration-driven sizing

Changing width, layers, heads, and context does not require rewriting the model.

## Fused QKV projection

More efficient than creating separate Python attention-head objects.

## Module boundaries

Tokenizer, data, model, trainer, checkpointing, and generation can evolve independently.

## Token-data artifacts

Text is tokenized once rather than during every training run.

## Checkpoint state

Optimizer and RNG state permit reliable continuation.

## Independent sampling generators

Evaluation does not perturb training batches.

## Weight tying

Reduces parameter count.

## Data-lineage hashes

Prevents silent incompatibility.

## Replaceable implementations

The educational BPE trainer could later be replaced by an optimized tokenizer while maintaining a similar external interface.

The explicit attention implementation could later be replaced by an optimized scaled-dot-product or Flash Attention kernel.

The stored `.pt` token stream could later be replaced with sharded memory-mapped storage.

Scalable design does not mean the current pure-Python implementation can train a production-scale LLM unchanged.

It means important boundaries have been placed so individual components can be upgraded without redesigning the entire repository.

---

# 30. Current technical limitations and future improvements

## 30.1 Educational BPE performance

The BPE trainer repeatedly scans the token sequence.

It is suitable for a modest corpus, not web-scale text.

Future replacement:

- Optimized native tokenizer
- Parallel pair counting
- Chunked training
- Better data structures

## 30.2 Character-level data splitting

The current 90/5/5 split can cut through a story.

A better pipeline would assign complete documents to train, validation, or test before concatenation.

## 30.3 `.pt` storage

Loading one complete token tensor is suitable for the current corpus.

Large-scale training should use:

- Binary shards
- Memory mapping
- Streaming
- Distributed samplers

## 30.4 Explicit attention matrix

Current attention materializes:

```text
[B,H,T,T]
```

This is transparent but scales quadratically with sequence length.

Future improvement:

- PyTorch scaled-dot-product attention
- Flash Attention
- Blockwise attention

## 30.5 No dropout

The current model has no dropout.

This keeps the architecture simple and generation deterministic under evaluation.

Dropout could be introduced if overfitting requires additional regularization.

## 30.6 Strict resume configuration

Resume currently requires exact training-config equality.

A future controlled extension mechanism could allow increasing `max_steps` while explicitly recalculating or preserving a schedule.

## 30.7 No gradient accumulation in full trainer

The concept was discussed but not integrated into the main training loop.

Future support could produce:

```text
effective batch size
=
micro-batch size
× accumulation steps
```

## 30.8 No structured logging file

Metrics are printed to the console.

Future improvements:

- JSON Lines metrics
- CSV logs
- TensorBoard
- Weights & Biases
- Experiment database

## 30.9 No early stopping

The best checkpoint is saved, but training does not stop automatically when validation fails to improve.

## 30.10 No KV cache

Generation recomputes prior context.

## 30.11 No model export format

Training checkpoints include optimizer state.

A smaller inference-only artifact could be exported later.

## 30.12 No formal test suite

Many diagnostic scripts should eventually become automated unit and integration tests.

---

# 31. What has been achieved

The complete base-language-model lifecycle now exists:

```text
Raw corpus                              Complete
Text normalization                     Complete
Train/validation/test split            Complete
Byte-level representation              Complete
Custom BPE training                    Complete
Tokenizer save/load                    Complete
EOS support                            Complete
Token dataset generation               Complete
Context-window sampling                Complete
Next-token targets                     Complete
Token embeddings                       Complete
RoPE                                   Complete
Q/K/V                                  Complete
Multi-head causal attention            Complete
RMSNorm                                Complete
SwiGLU                                 Complete
Residual connections                   Complete
Transformer stack                      Complete
Complete TinyGPT                       Complete
Weight tying                           Complete
Cross-entropy                          Complete
Backpropagation                        Complete
AdamW                                  Complete
Weight-decay groups                    Complete
Gradient clipping                      Complete
Warmup                                 Complete
Cosine decay                           Complete
Train/validation evaluation            Complete
Perplexity                             Complete
Tokens-per-second reporting            Complete
Checkpoint save                        Complete
Checkpoint load                        Complete
Best/latest distinction                Complete
Exact resume                           Complete
Autoregressive generation              Complete
Greedy decoding                        Complete
Temperature                            Complete
Top-k                                  Complete
Top-p                                  Complete
EOS stopping                           Complete
```

---

# 32. What remains before the final TinyChatGPT goal

The current model is TinyGPT, a base language model.

The final stated goal is TinyChatGPT, meaning a model that can participate in a user-assistant conversation.

The remaining major phases are:

## 32.1 Topic 21: Base-model evaluation

We need to compare:

```text
Untrained model
Latest checkpoint
Best checkpoint
```

using:

- Train loss
- Validation loss
- Test loss
- Perplexity
- Prompt continuations
- Next-token distributions
- Memorization checks
- Overfitting analysis

## 32.2 Instruction dataset

A chat model needs examples such as:

```text
User: What is rain?
Assistant: Rain is water that falls from clouds.
```

The dataset must be transformed into token sequences.

## 32.3 Chat template

The tokenizer/model need a consistent representation of roles.

Conceptually:

```text
<|system|>
You are a helpful assistant.
<|user|>
What is rain?
<|assistant|>
Rain is...
```

Special-token design must be handled carefully because modifying the tokenizer after base training changes vocabulary compatibility.

Possible approaches include:

- Reserve chat tokens before pretraining
- Extend vocabulary and initialize new rows
- Use text-based delimiters already representable by byte BPE

## 32.4 Supervised fine-tuning

The base model will be fine-tuned on instruction/response examples.

## 32.5 Assistant-only loss masking

During chat fine-tuning, the project should normally avoid training the model to predict every user token as if it were assistant output.

A mask can calculate loss only on assistant-response token positions.

## 32.6 Multi-turn context

Conversation history must be formatted and truncated to fit the context window.

## 32.7 Interactive CLI

The project can add:

```text
You:
TinyGPT:
```

with continuous conversation.

## 32.8 FastAPI serving

A local API can expose:

```text
POST /generate
POST /chat
```

## 32.9 Streaming

Generated tokens can be returned incrementally.

## 32.10 Inference engineering

Later improvements include:

- KV cache
- Quantization
- Batched inference
- CPU optimization
- Compiled execution
- GQA
- Flash Attention

## 32.11 Preference alignment

For conceptual completeness:

- Preference pairs
- Reward models
- DPO
- RLHF
- Safety evaluation

These are later alignment stages, not prerequisites for the first functioning TinyChatGPT.

---

# 33. Three-minute explanation for another developer

TinyGPT is a CPU-sized decoder-only Transformer built from scratch at the architecture level.

A TinyStories subset is downloaded into `data/raw/input.txt`. The data pipeline normalizes it and produces train, validation, and test text files. A custom byte-level BPE trainer starts with all 256 byte values and repeatedly merges frequent adjacent token pairs. It saves an ordered tokenizer artifact in `data/tokenizer/tokenizer.json`.

The frozen tokenizer encodes each story separately, adds EOS after every story, and stores each split as one long `torch.long` tensor. `TokenDataset` samples random context windows. The target is the same window shifted forward by one token.

The model maps `[B,T]` token IDs to `[B,T,C]` embeddings. Each Transformer block performs Pre-Norm multi-head causal attention followed by a Pre-Norm SwiGLU MLP, with residual connections around both sublayers. Attention uses a fused QKV projection, reshapes to `[B,H,T,D]`, applies RoPE to Queries and Keys, calculates scaled dot-product scores, masks future positions, applies softmax, aggregates Values, merges heads, and applies an output projection. Four independent blocks are stacked.

Final RMSNorm and a weight-tied language-model head produce `[B,T,V]` logits. Cross-entropy compares those logits with shifted target IDs after flattening batch and token dimensions. Backpropagation calculates gradients; gradient clipping controls unusually large norms; AdamW updates parameters with separate decay and no-decay groups. A warmup-plus-cosine schedule changes learning rate over time.

Training evaluates fixed train and validation batches, reports loss, perplexity, gradient norm, learning rate, and tokens per second. Checkpoints store model state, optimizer state, configuration, dataset/tokenizer hashes, progress, and RNG state. `latest.pt` is used for resume, while `best.pt` is used for generation.

Generation encodes a prompt, repeatedly runs TinyGPT on the visible context, takes the final-position vocabulary logits, applies greedy or sampled decoding, appends one token, stops on EOS when predicted, and decodes the full byte-level BPE token sequence into text.

The system is now a complete base language-model lifecycle. The next phase is evaluating what it actually learned, followed by instruction/chat fine-tuning.

---

# 34. Glossary

## Autoregressive

A generation process in which previously generated output becomes part of the next input.

## Batch

A group of independent training sequences processed together.

## BPE

Byte Pair Encoding. A tokenizer-training algorithm that repeatedly merges frequent adjacent token pairs.

## Buffer

A model-owned tensor that moves with the model but is not optimized.

## Causal mask

A mask that prevents a token from attending to future tokens.

## Checkpoint

A persistent snapshot of model, optimizer, progress, configuration, and RNG state.

## Context length

Maximum number of visible token positions in one model invocation.

## Cross-entropy

A loss that penalizes low probability assigned to the correct next token.

## Decoder-only Transformer

A Transformer architecture that predicts subsequent tokens using causal self-attention.

## Embedding

A trainable vector associated with a token ID.

## EOS

End-of-sequence or end-of-document token.

## Gradient

The local derivative of loss with respect to a parameter.

## Gradient clipping

Scaling gradients when their global norm exceeds a threshold.

## Head dimension

Feature width handled by one attention head.

## Hidden state

The model’s internal floating-point representation of a token in context.

## LM head

The final projection from hidden features to vocabulary logits.

## Logit

An unrestricted score for a vocabulary token before softmax.

## Module

A PyTorch neural-network component inheriting from `nn.Module`.

## Optimizer

An algorithm that changes parameters using gradients and optimizer state.

## Parameter

A trainable tensor owned by a neural-network module.

## Perplexity

`exp(loss)`, an uncertainty-related language-model metric.

## Pre-Norm

A Transformer arrangement in which normalization occurs before attention or MLP.

## Q/K/V

Query, Key, and Value projections used by attention.

## Residual connection

Addition of a sublayer output to the sublayer input.

## RMSNorm

Normalization based on root mean square rather than centered variance.

## RoPE

Rotary Position Embeddings. Position-dependent rotation applied to Queries and Keys.

## Softmax

A function that converts scores into a positive distribution summing to one.

## SwiGLU

A gated feed-forward network using SiLU activation.

## Token

The discrete unit processed by a language model.

## Token ID

An integer identifying one tokenizer vocabulary entry.

## Tokenizer

A reversible system mapping text to token IDs and token IDs back to text.

## Transformer block

A repeated model unit containing attention, an MLP, normalizations, and residual connections.

## Validation set

Held-out data used to monitor generalization without updating model weights.

## Vocabulary

The complete set of token IDs that the tokenizer and model understand.

## Weight tying

Sharing the token-embedding matrix with the output language-model head.

---

# 35. Current handoff state

A new developer joining now should understand that:

1. The environment and package layout are established.
2. The TinyStories subset provides a meaningful corpus for a very small model.
3. The tokenizer is custom byte-level BPE and must stay paired with model artifacts.
4. Train, validation, and test token streams are already distinct.
5. The model is a modernized decoder-only architecture using RMSNorm, RoPE, fused multi-head attention, SwiGLU, residual connections, and weight tying.
6. The model predicts next tokens through `[B,T,V]` logits.
7. Cross-entropy, backpropagation, AdamW, gradient clipping, warmup, and cosine decay form the optimization system.
8. Validation loss determines `best.pt`.
9. `latest.pt` exists for exact continuation.
10. Generation uses the trained base model autoregressively.
11. The current system is a base TinyGPT, not yet an instruction-following TinyChatGPT.
12. Topic 21 should evaluate what the base model actually learned before chat fine-tuning begins.