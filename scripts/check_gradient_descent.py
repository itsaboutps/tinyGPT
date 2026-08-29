import torch


w = torch.tensor(
    2.0,
    requires_grad=True,
)


loss = w ** 2


print("Parameter before:")
print(w.item())


print()
print("Loss:")
print(loss.item())


loss.backward()


print()
print("Gradient:")
print(w.grad.item())


learning_rate = 0.1


with torch.no_grad():
    w -= (
        learning_rate
        * w.grad
    )


print()
print("Parameter after:")
print(w.item())