import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.mlp import (
    SwiGLU,
)
from tinygpt.model.norm import (
    RMSNorm,
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


norm = RMSNorm(
    d_model=config.d_model,
    eps=config.rms_norm_eps,
)


mlp = SwiGLU(
    d_model=config.d_model,
    d_ff=config.d_ff,
)


normalized = norm(x)


mlp_output = mlp(
    normalized
)


output = (
    x
    +
    mlp_output
)


print("=" * 60)
print("PRE-NORM SWIGLU + RESIDUAL")
print("=" * 60)


print()
print("Input:")
print(x.shape)


print()
print("Normalized:")
print(normalized.shape)


print()
print("MLP output:")
print(mlp_output.shape)


print()
print("Residual output:")
print(output.shape)