import torch

from tinygpt.model.embeddings import (
    TokenEmbedding,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.utils.random import (
    set_seed,
)
from tinygpt.utils.parameters import (
    count_parameters,
    count_trainable_parameters,
)


set_seed(42)


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


d_model = 8


embedding = TokenEmbedding(
    vocab_size=tokenizer.vocab_size,
    d_model=d_model,
)


print("=" * 60)
print("TOKEN EMBEDDING")
print("=" * 60)


print()
print("Vocabulary size:")
print(tokenizer.vocab_size)


print()
print("Embedding dimension:")
print(d_model)


print()
print("Embedding weight shape:")
print(
    embedding.embedding.weight.shape
)


text = "hello"


token_ids = tokenizer.encode(
    text
)


x = torch.tensor(
    [token_ids],
    dtype=torch.long,
)


print()
print("Text:")
print(repr(text))


print()
print("Token IDs:")
print(x)


print()
print("Token ID shape:")
print(x.shape)


embedded = embedding(x)


print()
print("Embedding output shape:")
print(embedded.shape)


print()
print("Embedding output:")
print(embedded)

print()
print("First embedding row:")
print(
    embedding.embedding.weight[0]
)


print()
print("Row 1:")
print(
    embedding.embedding.weight[1]
)

first_token_id = x[0, 0]


looked_up_directly = (
    embedding.embedding.weight[
        first_token_id
    ]
)


looked_up_through_forward = (
    embedded[0, 0]
)


print()
print("First token ID:")
print(first_token_id)


print()
print("Direct table lookup:")
print(
    looked_up_directly
)


print()
print("Forward result:")
print(
    looked_up_through_forward
)


print()
print("Exactly equal:")
print(
    torch.equal(
        looked_up_directly,
        looked_up_through_forward,
    )
)

for name, parameter in (
    embedding.named_parameters()
):
    print(
        name,
        parameter.shape,
        parameter.requires_grad,
    )
    

print()
print("Total parameters:")
print(
    count_parameters(
        embedding
    )
)


print()
print("Trainable parameters:")
print(
    count_trainable_parameters(
        embedding
    )
)