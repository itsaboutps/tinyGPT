import torch

from tinygpt.model.embeddings import (
    LearnedPositionEmbedding,
    TokenEmbedding,
)


B = 2
T = 4
C = 8
VOCAB_SIZE = 20


token_ids = torch.tensor(
    [
        [4, 7, 4, 9],
        [4, 7, 4, 9],
    ],
    dtype=torch.long,
)


token_embedding = TokenEmbedding(
    vocab_size=VOCAB_SIZE,
    d_model=C,
)


position_embedding = (
    LearnedPositionEmbedding(
        context_length=16,
        d_model=C,
    )
)


token_vectors = token_embedding(
    token_ids
)


position_vectors = position_embedding(
    sequence_length=T,
    device=token_ids.device,
)


x = (
    token_vectors
    + position_vectors
)


print("Token IDs:")
print(token_ids.shape)


print()
print("Token vectors:")
print(token_vectors.shape)


print()
print("Position vectors:")
print(position_vectors.shape)


print()
print("Combined:")
print(x.shape)


print()
print(
    "Same token ID at position 0 and 2?"
)

print(
    token_ids[0, 0]
    ==
    token_ids[0, 2]
)


print()
print(
    "Raw token embeddings equal?"
)

print(
    torch.equal(
        token_vectors[0, 0],
        token_vectors[0, 2],
    )
)


print()
print(
    "After position addition equal?"
)

print(
    torch.equal(
        x[0, 0],
        x[0, 2],
    )
)