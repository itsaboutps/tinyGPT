import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.transformer import (
    TransformerStack,
)
from tinygpt.utils.parameters import (
    count_trainable_parameters,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


config = ModelConfig(
    vocab_size=1024
)


stack = TransformerStack(
    config
)


B = 2
T = 6


x = torch.randn(
    B,
    T,
    config.d_model,
)


output = stack(
    x
)


print("=" * 60)
print("TRANSFORMER STACK")
print("=" * 60)


print()
print("Number of blocks:")
print(
    len(stack.blocks)
)


print()
print("Input:")
print(x.shape)


print()
print("Output:")
print(output.shape)


print()
print(
    "Trainable parameters:"
)

print(
    count_trainable_parameters(
        stack
    )
)


same_weights = torch.equal(
    stack
    .blocks[0]
    .attention
    .qkv_projection
    .weight,

    stack
    .blocks[1]
    .attention
    .qkv_projection
    .weight,
)


print()
print(
    "Block 0 and Block 1 "
    "share identical QKV weights:"
)

print(
    same_weights
)