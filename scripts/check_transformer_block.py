import torch

from tinygpt.config import ModelConfig
from tinygpt.model.block import (
    TransformerBlock,
)
from tinygpt.utils.parameters import (
    count_trainable_parameters,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


config = ModelConfig(
    vocab_size=1024,
)


B = 2
T = 6


x = torch.randn(
    B,
    T,
    config.d_model,
)


block = TransformerBlock(
    config=config
)


output = block(
    x
)


print("=" * 60)
print("TRANSFORMER BLOCK")
print("=" * 60)


print()
print("Input shape:")
print(x.shape)


print()
print("Output shape:")
print(output.shape)


print()
print("Shape preserved:")
print(
    x.shape
    ==
    output.shape
)


print()
print("Trainable parameters:")
print(
    count_trainable_parameters(
        block
    )
)