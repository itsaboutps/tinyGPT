import torch

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
from tinygpt.training.optimizer import (
    create_optimizer,
)
from tinygpt.training.step import (
    train_step,
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


optimizer = create_optimizer(
    model=model,
    config=training_config,
)


x, y = dataset.get_batch(
    batch_size=2,
    context_length=32,
    device=device,
)


parameter = (
    model
    .transformer
    .blocks[0]
    .attention
    .qkv_projection
    .weight
)


before = (
    parameter
    .detach()
    .clone()
)


metrics = train_step(
    model=model,
    optimizer=optimizer,
    x=x,
    y=y,
    grad_clip_norm=(
        training_config
        .grad_clip_norm
    ),
)


after = (
    parameter
    .detach()
    .clone()
)


changed = (
    not torch.equal(
        before,
        after,
    )
)


maximum_change = (
    after
    .sub(before)
    .abs()
    .max()
    .item()
)


print("=" * 60)
print("FIRST REAL TINYGPT UPDATE")
print("=" * 60)


print()
print("Loss:")
print(
    metrics["loss"]
)


print()
print("Gradient norm:")
print(
    metrics[
        "gradient_norm"
    ]
)


print()
print(
    "Parameter changed:"
)

print(
    changed
)


print()
print(
    "Maximum absolute "
    "parameter change:"
)

print(
    maximum_change
)