import torch


head_dim = 8
base = 10000.0


dimension_indices = torch.arange(
    0,
    head_dim,
    2,
    dtype=torch.float32,
)


inverse_frequencies = 1.0 / (
    base
    ** (
        dimension_indices
        / head_dim
    )
)


print("Dimension indices:")
print(dimension_indices)


print()
print("Inverse frequencies:")
print(inverse_frequencies)



sequence_length = 5


positions = torch.arange(
    sequence_length,
    dtype=torch.float32,
)


angles = (
    positions[:, None]
    *
    inverse_frequencies[None, :]
)


print()
print("Positions:")
print(positions)


print()
print("Angles shape:")
print(angles.shape)


print()
print("Angles:")
print(angles)


cos_values = torch.cos(
    angles
)

sin_values = torch.sin(
    angles
)


print()
print("Cosine:")
print(cos_values)


print()
print("Sine:")
print(sin_values)