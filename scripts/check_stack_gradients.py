import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.transformer import (
    TransformerStack,
)
from tinygpt.utils.random import (
    set_seed,
)


set_seed(42)


config = ModelConfig(
    vocab_size=1024
)


stack = TransformerStack(
    config
)


x = torch.randn(
    2,
    6,
    config.d_model,
    requires_grad=True,
)


output = stack(
    x
)


fake_loss = (
    output.pow(2).mean()
)


fake_loss.backward()


print(
    "Input gradient exists:"
)

print(
    x.grad is not None
)


for index, block in enumerate(
    stack.blocks
):

    grad = (
        block
        .attention
        .qkv_projection
        .weight
        .grad
    )

    print(
        f"Block {index} "
        f"QKV gradient exists:",
        grad is not None,
    )