import math

import torch

from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.training.loss import (
    language_model_loss,
)


def evaluate_loss(
    model: torch.nn.Module,
    dataset: TokenDataset,
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> dict:

    was_training = model.training

    model.eval()


    generator = (
        torch.Generator()
        .manual_seed(seed)
    )


    losses = []


    with torch.no_grad():

        for _ in range(
            num_batches
        ):

            x, y = dataset.get_batch(
                batch_size=batch_size,
                context_length=(
                    context_length
                ),
                device=device,
                generator=generator,
            )


            logits = model(
                x
            )


            loss = language_model_loss(
                logits=logits,
                targets=y,
            )


            losses.append(
                loss.item()
            )


    mean_loss = (
        sum(losses)
        / len(losses)
    )


    perplexity = math.exp(
        mean_loss
    )


    if was_training:
        model.train()


    return {
        "loss": mean_loss,
        "perplexity": perplexity,
    }