import torch

from tinygpt.sft.dataset import (
    IGNORE_INDEX,
)
from tinygpt.sft.loss import (
    sft_loss,
)


def sft_backward_micro_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict:

    model.train()


    logits = model(
        input_ids,
        attention_mask=attention_mask,
    )


    loss_sum = sft_loss(
        logits=logits,
        targets=targets,
        reduction="sum",
    )


    active_tokens = (
        targets
        != IGNORE_INDEX
    ).sum().item()


    if active_tokens <= 0:
        raise ValueError(
            "Micro-batch contains no "
            "active assistant targets"
        )


    loss_sum.backward()


    mean_loss = (
        loss_sum.item()
        / active_tokens
    )


    return {
        "loss_sum": (
            loss_sum.item()
        ),
        "mean_loss": (
            mean_loss
        ),
        "active_tokens": (
            active_tokens
        ),
    }


def apply_sft_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_active_tokens: int,
    grad_clip_norm: float,
) -> dict:

    if total_active_tokens <= 0:
        raise ValueError(
            "total_active_tokens must "
            "be greater than 0"
        )


    #
    # We accumulated gradients from
    # summed token losses.
    #
    # Convert them into gradients of
    # mean loss across all active tokens.
    #

    scale = (
        1.0
        / total_active_tokens
    )


    for parameter in model.parameters():

        if parameter.grad is not None:

            parameter.grad.mul_(
                scale
            )


    gradient_norm = (
        torch.nn.utils
        .clip_grad_norm_(
            model.parameters(),
            max_norm=(
                grad_clip_norm
            ),
        )
    )


    optimizer.step()


    optimizer.zero_grad(
        set_to_none=True
    )


    return {
        "gradient_norm": (
            gradient_norm.item()
        ),
    }