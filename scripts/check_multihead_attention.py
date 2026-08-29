import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.attention import (
    MultiHeadCausalSelfAttention,
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


output, weights = attention(
    x,
    return_attention=True,
)


print("=" * 60)
print("MULTI-HEAD CAUSAL SELF-ATTENTION")
print("=" * 60)


print()
print("Input:")
print(x.shape)


print()
print("Heads:")
print(config.n_heads)


print()
print("Head dimension:")
print(config.head_dim)


print()
print("Attention weights:")
print(weights.shape)


print()
print("Output:")
print(output.shape)


print()
print("Trainable parameters:")
print(
    count_trainable_parameters(
        attention
    )
)

future_mask = torch.triu(
    torch.ones(
        T,
        T,
        dtype=torch.bool,
    ),
    diagonal=1,
)


future_values = (
    weights[
        :,
        :,
        future_mask
    ]
)


print()
print(
    "Maximum future attention:"
)

print(
    future_values.abs().max()
)

row_sums = (
    weights.sum(dim=-1)
)


print()
print(
    "All rows sum to one?"
)

print(
    torch.allclose(
        row_sums,
        torch.ones_like(
            row_sums
        ),
        atol=1e-5,
    )
)