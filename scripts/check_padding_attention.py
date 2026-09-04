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


def main():

    set_seed(42)


    config = ModelConfig(
        vocab_size=100,
    )


    model = TinyGPT(
        config
    )

    model.eval()


    real_tokens = [
        10,
        20,
        30,
        40,
    ]


    x1 = torch.tensor(
        [[
            10,
            20,
            30,
            40,
            1,
            1,
        ]],
        dtype=torch.long,
    )


    x2 = torch.tensor(
        [[
            10,
            20,
            30,
            40,
            75,
            88,
        ]],
        dtype=torch.long,
    )


    mask = torch.tensor(
        [[
            True,
            True,
            True,
            True,
            False,
            False,
        ]]
    )


    with torch.no_grad():

        logits_1 = model(
            x1,
            attention_mask=mask,
        )

        logits_2 = model(
            x2,
            attention_mask=mask,
        )


    same_real_logits = torch.allclose(
        logits_1[
            :,
            :4,
            :
        ],
        logits_2[
            :,
            :4,
            :
        ],
        atol=1e-6,
    )


    print(
        "Real-position logits unchanged:",
        same_real_logits,
    )


if __name__ == "__main__":
    main()