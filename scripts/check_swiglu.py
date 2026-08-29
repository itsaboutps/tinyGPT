import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.mlp import (
    SwiGLU,
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


B = 2
T = 6


x = torch.randn(
    B,
    T,
    config.d_model,
)


mlp = SwiGLU(
    d_model=config.d_model,
    d_ff=config.d_ff,
)


output = mlp(
    x
)


print("=" * 60)
print("SWIGLU MLP")
print("=" * 60)


print()
print("Input:")
print(x.shape)


print()
print("Hidden dimension:")
print(config.d_ff)


print()
print("Output:")
print(output.shape)


print()
print("Trainable parameters:")
print(
    count_trainable_parameters(
        mlp
    )
)