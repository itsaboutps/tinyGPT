import torch

from tinygpt.model.embeddings import (
    TokenEmbedding,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


embedding = TokenEmbedding(
    vocab_size=10,
    d_model=4,
)


token_ids = torch.tensor(
    [
        [2, 5, 2]
    ],
    dtype=torch.long,
)


output = embedding(
    token_ids
)


print("Output shape:")
print(output.shape)


loss = output.sum()


print()
print("Fake loss:")
print(loss)


loss.backward()


print()
print("Embedding gradients:")
print(
    embedding.embedding.weight.grad
)