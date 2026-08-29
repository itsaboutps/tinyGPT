import torch
import torch.nn.functional as F


logits = torch.tensor(
    [
        [1.2, -0.5, 2.8, 0.3, 1.0]
    ],
    dtype=torch.float32,
)


target = torch.tensor(
    [2],
    dtype=torch.long,
)


loss = F.cross_entropy(
    logits,
    target,
)


probabilities = torch.softmax(
    logits,
    dim=-1,
)


print("=" * 60)
print("CROSS ENTROPY")
print("=" * 60)


print()
print("Logits:")
print(logits)


print()
print("Probabilities:")
print(probabilities)


print()
print(
    "Probability of correct token:"
)

print(
    probabilities[0, 2]
)


print()
print("Loss:")
print(loss)