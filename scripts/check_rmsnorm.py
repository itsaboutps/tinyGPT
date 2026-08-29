import torch

from tinygpt.model.norm import (
    RMSNorm,
)
from tinygpt.utils.parameters import (
    count_trainable_parameters,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


B = 2
T = 4
C = 8


x = torch.randn(
    B,
    T,
    C,
) * 10


norm = RMSNorm(
    d_model=C,
    eps=1e-5,
)


output = norm(x)


print("=" * 60)
print("RMSNORM")
print("=" * 60)


print()
print("Input shape:")
print(x.shape)


print()
print("Output shape:")
print(output.shape)


print()
print("Trainable parameters:")
print(
    count_trainable_parameters(
        norm
    )
)


print()
print("Learned weight:")
print(norm.weight)


print()
print("First token before:")
print(x[0, 0])


print()
print("First token after:")
print(output[0, 0])




output_rms = torch.sqrt(
    output.pow(2).mean(
        dim=-1
    )
)


print()
print(
    "Output RMS values:"
)

print(
    output_rms
)


loss = output.sum()

loss.backward()


print()
print(
    "RMSNorm weight gradient:"
)

print(
    norm.weight.grad
)