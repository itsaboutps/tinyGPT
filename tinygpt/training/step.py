import torch

from tinygpt.training.loss import (
    language_model_loss,
)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip_norm: float,
) -> dict:

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )


    logits = model(
        x
    )


    loss = language_model_loss(
        logits=logits,
        targets=y,
    )


    loss.backward()


    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
        )
    )


    optimizer.step()


    return {
        "loss": loss.item(),
        "gradient_norm": (
            gradient_norm.item()
        ),
    }