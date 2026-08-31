import torch

from tinygpt.config import (
    TrainingConfig,
)


def create_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
):

    decay_params = []

    no_decay_params = []


    for parameter in model.parameters():

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


    parameter_groups = [
        {
            "params": decay_params,
            "weight_decay": (
                config.weight_decay
            ),
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]


    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
    )


    return optimizer