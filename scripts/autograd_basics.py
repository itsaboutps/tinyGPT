import torch


x = torch.tensor(
    3.0,
    requires_grad=True
)

print("x:")
print(x)


y = x * x

print()
print("y = x * x:")
print(y)


y.backward()


print()
print("Gradient of y with respect to x:")
print(x.grad)


# forward pass
#     ↓
# result
#     ↓
# backward pass
#     ↓
# gradient

# which will Become
# GPT forward pass
#     ↓
# loss
#     ↓
# backward()
#     ↓
# millions of gradients