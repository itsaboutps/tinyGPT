import torch

from tinygpt.utils.random import set_seed


set_seed(42)

first = torch.randn(5)

print("First tensor:")
print(first)


set_seed(42)

second = torch.randn(5)

print()

print("Second tensor:")
print(second)


print()

print("Are they equal?")
print(torch.equal(first, second))