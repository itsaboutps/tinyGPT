import torch

from tinygpt.config import ModelConfig
from tinygpt.model.block import (
    TransformerBlock,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


config = ModelConfig(
    vocab_size=1024
)


block = TransformerBlock(
    config
)


x = torch.randn(
    2,
    6,
    config.d_model,
    requires_grad=True,
)


output = block(
    x
)


fake_loss = (
    output.pow(2).mean()
)


print("Fake loss:")
print(fake_loss)


fake_loss.backward()


print()
print(
    "Input gradient exists:"
)

print(
    x.grad is not None
)


print()
print(
    "Input gradient shape:"
)

print(
    x.grad.shape
)


print()
print(
    "QKV gradient exists:"
)

print(
    block
    .attention
    .qkv_projection
    .weight
    .grad
    is not None
)


print()
print(
    "MLP gate gradient exists:"
)

print(
    block
    .mlp
    .gate_projection
    .weight
    .grad
    is not None
)


print()
print(
    "Attention norm gradient exists:"
)

print(
    block
    .attention_norm
    .weight
    .grad
    is not None
)