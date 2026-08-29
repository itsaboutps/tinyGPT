import torch

from tinygpt.config import ModelConfig
from tinygpt.model.gpt import TinyGPT
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.utils.parameters import (
    count_parameters,
    count_trainable_parameters,
)
from tinygpt.utils.random import set_seed

from tinygpt.utils.parameters import (
    count_parameters,
    count_trainable_parameters,
    parameter_size_mb,
)


set_seed(42)


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


config = ModelConfig(
    vocab_size=tokenizer.vocab_size
)


model = TinyGPT(
    config
)


B = 2
T = 16


token_ids = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(B, T),
    dtype=torch.long,
)


logits = model(
    token_ids
)


print("=" * 60)
print("TINYGPT")
print("=" * 60)


print()
print("Vocabulary size:")
print(config.vocab_size)


print()
print("Input shape:")
print(token_ids.shape)


print()
print("Logits shape:")
print(logits.shape)


print()
print(
    "Expected logits shape:"
)

print(
    (
        B,
        T,
        config.vocab_size,
    )
)


print()
print("Total parameters:")
print(
    count_parameters(
        model
    )
)


print()
print(
    "Trainable parameters:"
)

print(
    count_trainable_parameters(
        model
    )
)


print()
print(
    "Logits for first sequence, "
    "first position:"
)

print(
    logits[0, 0]
)


print()
print(
    "Number of logits:"
)

print(
    logits[0, 0].numel()
)



probabilities = torch.softmax(
    logits,
    dim=-1,
)


first_position_probs = (
    probabilities[0, 0]
)


top_probability, top_token_id = (
    torch.max(
        first_position_probs,
        dim=-1,
    )
)


print()
print("Top token ID:")
print(
    top_token_id.item()
)


print()
print("Probability:")
print(
    top_probability.item()
)


print()
print("Decoded token:")
print(
    repr(
        tokenizer.decode(
            [
                top_token_id.item()
            ]
        )
    )
)


print()
print(
    "Parameter memory MB:"
)

print(
    round(
        parameter_size_mb(
            model
        ),
        3,
    )
)