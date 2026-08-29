import torch

from tinygpt.model.rope import (
    RotaryPositionEmbedding,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


B = 2
H = 4
T = 6
D = 8


q = torch.randn(
    B,
    H,
    T,
    D,
)


rope = RotaryPositionEmbedding(
    head_dim=D
)


rotated_q = rope(q)


print("=" * 60)
print("ROTARY POSITION EMBEDDING")
print("=" * 60)


print()
print("Input shape:")
print(q.shape)


print()
print("Output shape:")
print(rotated_q.shape)


print()
print(
    "Shape preserved?"
)

print(
    q.shape
    ==
    rotated_q.shape
)


print()
print("Position 0 before:")
print(
    q[0, 0, 0]
)


print()
print("Position 0 after:")
print(
    rotated_q[0, 0, 0]
)


print()
print(
    "Position 0 unchanged?"
)

print(
    torch.allclose(
        q[0, 0, 0],
        rotated_q[0, 0, 0],
    )
)


print()
print("Position 1 before:")
print(
    q[0, 0, 1]
)


print()
print("Position 1 after:")
print(
    rotated_q[0, 0, 1]
)


print()
print(
    "Position 1 changed?"
)

print(
    not torch.allclose(
        q[0, 0, 1],
        rotated_q[0, 0, 1],
    )
)


before_norm = torch.linalg.vector_norm(
    q,
    dim=-1,
)


after_norm = torch.linalg.vector_norm(
    rotated_q,
    dim=-1,
)


print()
print(
    "Vector magnitudes preserved?"
)

print(
    torch.allclose(
        before_norm,
        after_norm,
        atol=1e-5,
    )
)