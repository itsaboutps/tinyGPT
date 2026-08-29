import torch
import torch.nn.functional as F


x = torch.tensor(
    [
        -5.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        5.0,
    ]
)


y = F.silu(x)


print("=" * 60)
print("SILU")
print("=" * 60)


print()
print("Input:")
print(x)


print()
print("Output:")
print(y)