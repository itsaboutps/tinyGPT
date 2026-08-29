import torch
import torch.nn as nn

from tinygpt.utils.random import set_seed


set_seed(42)


B = 2
T = 4
C = 8
D = 4


x = torch.randn(
    B,
    T,
    C,
)


query_projection = nn.Linear(
    in_features=C,
    out_features=D,
    bias=False,
)


q = query_projection(x)


print("=" * 60)
print("QUERY PROJECTION")
print("=" * 60)


print()
print("Input shape:")
print(x.shape)


print()
print("Projection weight shape:")
print(
    query_projection.weight.shape
)


print()
print("Query shape:")
print(q.shape)


key_projection = nn.Linear(
    in_features=C,
    out_features=D,
    bias=False,
)


value_projection = nn.Linear(
    in_features=C,
    out_features=D,
    bias=False,
)


q = query_projection(x)
k = key_projection(x)
v = value_projection(x)


print()
print("Q:")
print(q.shape)


print()
print("K:")
print(k.shape)


print()
print("V:")
print(v.shape)


scores = (
    q
    @
    k.transpose(-2, -1)
)


print()
print("Attention score shape:")
print(scores.shape)


print()
print("Attention scores:")
print(scores)


scale = D ** -0.5


scaled_scores = (
    scores * scale
)


print()
print("Scale:")
print(scale)


print()
print("Scaled scores:")
print(scaled_scores)


T = 5


mask = torch.tril(
    torch.ones(
        T,
        T,
        dtype=torch.bool,
    )
)


print(mask)