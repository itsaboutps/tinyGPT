import math

import torch

from tinygpt.config import (
    TrainingConfig,
)


def get_learning_rate(
    step: int,
    config: TrainingConfig,
) -> float:

    if step < 0:
        raise ValueError(
            "step cannot be negative"
        )

    if step >= config.max_steps:
        raise ValueError(
            "step exceeds max_steps"
        )


    if (
        config.warmup_steps > 0
        and step < config.warmup_steps
    ):
        return (
            config.learning_rate
            * (step + 1)
            / config.warmup_steps
        )


    decay_steps = (
        config.max_steps
        - config.warmup_steps
    )


    if decay_steps <= 1:
        return (
            config.min_learning_rate
        )


    decay_step = (
        step
        - config.warmup_steps
    )


    progress = (
        decay_step
        / (decay_steps - 1)
    )


    progress = min(
        max(progress, 0.0),
        1.0,
    )


    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


    learning_rate = (
        config.min_learning_rate
        +
        cosine
        * (
            config.learning_rate
            - config.min_learning_rate
        )
    )


    return learning_rate


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:

    for group in optimizer.param_groups:

        group["lr"] = (
            learning_rate
        )