import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.gpt import (
    TinyGPT,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


config = ModelConfig(
    vocab_size=1024
)


model = TinyGPT(
    config
)


model.eval()


sequence_a = torch.tensor(
    [
        [
            10,
            20,
            30,
            40,
            50,
        ]
    ],
    dtype=torch.long,
)


sequence_b = torch.tensor(
    [
        [
            10,
            20,
            30,
            400,
            500,
        ]
    ],
    dtype=torch.long,
)


with torch.no_grad():

    logits_a = model(
        sequence_a
    )

    logits_b = model(
        sequence_b
    )


same_prefix_logits = (
    torch.allclose(
        logits_a[:, :3],
        logits_b[:, :3],
        atol=1e-6,
    )
)


print(
    "Prefix logits identical:"
)

print(
    same_prefix_logits
)


print()
print(
    "Future positions differ:"
)

print(
    not torch.allclose(
        logits_a[:, 3:],
        logits_b[:, 3:],
    )
)