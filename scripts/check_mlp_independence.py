import torch

from tinygpt.model.mlp import (
    SwiGLU,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


mlp = SwiGLU(
    d_model=8,
    d_ff=24,
)


x = torch.randn(
    1,
    3,
    8,
)


original_output = mlp(
    x
)


modified_x = x.clone()


modified_x[
    0,
    0,
] = modified_x[
    0,
    0,
] + 100.0


modified_output = mlp(
    modified_x
)


print(
    "Position 0 changed:"
)

print(
    not torch.allclose(
        original_output[0, 0],
        modified_output[0, 0],
    )
)


print()
print(
    "Position 1 unchanged:"
)

print(
    torch.allclose(
        original_output[0, 1],
        modified_output[0, 1],
    )
)


print()
print(
    "Position 2 unchanged:"
)

print(
    torch.allclose(
        original_output[0, 2],
        modified_output[0, 2],
    )
)