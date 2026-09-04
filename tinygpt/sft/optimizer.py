import torch

from tinygpt.config import (
    SFTConfig,
)


def create_sft_optimizer(
    model: torch.nn.Module,
    config: SFTConfig,
) -> torch.optim.Optimizer:

    decay_params = []

    no_decay_params = []


    for parameter in (
        model.parameters()
    ):

        if not parameter.requires_grad:
            continue


        if parameter.ndim >= 2:

            decay_params.append(
                parameter
            )

        else:

            no_decay_params.append(
                parameter
            )


    groups = [
        {
            "params": (
                decay_params
            ),
            "weight_decay": (
                config.weight_decay
            ),
        },
        {
            "params": (
                no_decay_params
            ),
            "weight_decay": 0.0,
        },
    ]


    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
    )