

import torch

from tinygpt.utils.device import get_device


device = get_device()

print("PyTorch version:")
print(torch.__version__)

print()

print("Selected device:")
print(device)

x = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)

x = x.to(device)

print()

print("Tensor:")
print(x)

print()

print("Tensor device:")
print(x.device)