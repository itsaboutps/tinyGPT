import torch

from tinygpt.sft.loss import (
    sft_loss,
)


def sft_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_clip_norm: float,
) -> dict:

    model.train()


    optimizer.zero_grad(
        set_to_none=True
    )


    logits = model(
        input_ids,
        attention_mask=(
            attention_mask
        ),
    )


    loss = sft_loss(
        logits=logits,
        targets=targets,
    )


    loss.backward()


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


    return {
        "loss": (
            loss.item()
        ),
        "gradient_norm": (
            gradient_norm.item()
        ),
    }