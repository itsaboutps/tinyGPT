from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.model.gpt import TinyGPT
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.training.loss import (
    language_model_loss,
)
from tinygpt.utils.device import (
    get_device,
)
from tinygpt.utils.random import (
    set_seed,
)


training_config = TrainingConfig()


set_seed(
    training_config.seed
)


device = get_device()


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


model_config = ModelConfig(
    vocab_size=tokenizer.vocab_size
)


dataset = TokenDataset(
    "data/tokens/train.pt"
)


model = TinyGPT(
    model_config
).to(device)


x, y = dataset.get_batch(
    batch_size=2,
    context_length=32,
    device=device,
)


logits = model(
    x
)


loss = language_model_loss(
    logits,
    y,
)


loss.backward()


print("=" * 60)
print("REAL LANGUAGE GRADIENTS")
print("=" * 60)


print()
print("Loss:")
print(loss.item())


print()
print(
    "Embedding gradient exists:"
)

print(
    model
    .token_embedding
    .embedding
    .weight
    .grad
    is not None
)


print()
print(
    "Block 0 QKV gradient exists:"
)

print(
    model
    .transformer
    .blocks[0]
    .attention
    .qkv_projection
    .weight
    .grad
    is not None
)


print()
print(
    "Block 3 MLP gradient exists:"
)

print(
    model
    .transformer
    .blocks[3]
    .mlp
    .gate_projection
    .weight
    .grad
    is not None
)


print()
print(
    "Final norm gradient exists:"
)

print(
    model
    .final_norm
    .weight
    .grad
    is not None
)


gradient = (
    model
    .transformer
    .blocks[0]
    .attention
    .qkv_projection
    .weight
    .grad
)


print()
print("QKV weight shape:")
print(
    model
    .transformer
    .blocks[0]
    .attention
    .qkv_projection
    .weight
    .shape
)


print()
print("QKV gradient shape:")
print(
    gradient.shape
)


gradient = (
    model
    .transformer
    .blocks[0]
    .attention
    .qkv_projection
    .weight
    .grad
)


print()
print("QKV weight shape:")
print(
    model
    .transformer
    .blocks[0]
    .attention
    .qkv_projection
    .weight
    .shape
)


print()
print("QKV gradient shape:")
print(
    gradient.shape
)

import torch.nn.functional as F


B, T, V = logits.shape


per_token_loss = F.cross_entropy(
    logits.reshape(-1, V),
    y.reshape(-1),
    reduction="none",
)


per_token_loss = (
    per_token_loss.reshape(
        B,
        T,
    )
)


print()
print(
    "Per-token loss shape:"
)

print(
    per_token_loss.shape
)


print()
print(
    "First sequence losses:"
)

print(
    per_token_loss[0]
)