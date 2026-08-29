import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.attention import (
    CausalSelfAttentionHead,
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
C = config.d_model


x = torch.randn(
    B,
    T,
    C,
)


attention = (
    CausalSelfAttentionHead(
        d_model=config.d_model,
        head_dim=config.head_dim,
        context_length=(
            config.context_length
        ),
        rope_base=(
            config.rope_base
        ),
    )
)


output = attention(x)


print("=" * 60)
print("CAUSAL SELF-ATTENTION HEAD")
print("=" * 60)


print()
print("Input shape:")
print(x.shape)


print()
print("Head dimension:")
print(config.head_dim)


print()
print("Output shape:")
print(output.shape)


print()
print("Trainable parameters:")
print(
    count_trainable_parameters(
        attention
    )
)

output, weights = attention(
    x,
    return_attention=True,
)

print()
print("Attention weights shape:")
print(weights.shape)


print()
print(
    "Attention weights "
    "for first sequence:"
)

print(weights[0])



future_mask = torch.triu(
    torch.ones(
        T,
        T,
        dtype=torch.bool,
    ),
    diagonal=1,
)


future_weights = (
    weights[:, future_mask]
)


print()
print(
    "Maximum future attention:"
)

print(
    future_weights.abs().max()
)

row_sums = (
    weights.sum(dim=-1)
)


print()
print(
    "Attention row sums:"
)

print(row_sums)