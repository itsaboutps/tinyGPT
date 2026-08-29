import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.attention import (
    MultiHeadCausalSelfAttention,
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


attention = (
    MultiHeadCausalSelfAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        context_length=(
            config.context_length
        ),
        rope_base=config.rope_base,
    )
)


normalized = norm(x)


attention_output = attention(
    normalized
)


output = (
    x
    +
    attention_output
)


print("=" * 60)
print("PRE-NORM ATTENTION + RESIDUAL")
print("=" * 60)


print()
print("Input:")
print(x.shape)


print()
print("After RMSNorm:")
print(normalized.shape)


print()
print("Attention output:")
print(attention_output.shape)


print()
print("Residual output:")
print(output.shape)