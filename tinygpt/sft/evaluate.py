import math

import torch

from tinygpt.sft.dataloader import (
    create_sft_dataloader,
)
from tinygpt.sft.dataset import (
    IGNORE_INDEX,
    SFTDataset,
)
from tinygpt.sft.loss import (
    sft_loss,
)


@torch.inference_mode()
def evaluate_sft_loss(
    model: torch.nn.Module,
    dataset: SFTDataset,
    tokenizer,
    batch_size: int,
    device: torch.device,
) -> dict:

    if len(dataset) == 0:
        raise ValueError(
            "SFT evaluation dataset is empty"
        )


    loader = create_sft_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        pad_token_id=(
            tokenizer.eos_token_id
        ),
        shuffle=False,
    )


    was_training = (
        model.training
    )


    model.eval()


    total_loss = 0.0

    total_active_tokens = 0


    for batch in loader:

        input_ids = (
            batch.input_ids
            .to(device)
        )


        targets = (
            batch.targets
            .to(device)
        )


        attention_mask = (
            batch.attention_mask
            .to(device)
        )


        logits = model(
            input_ids,
            attention_mask=(
                attention_mask
            ),
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


        total_loss += (
            loss_sum.item()
        )


        total_active_tokens += (
            active_tokens
        )


    if total_active_tokens == 0:
        raise ValueError(
            "No active SFT targets "
            "were evaluated"
        )


    mean_loss = (
        total_loss
        / total_active_tokens
    )


    perplexity = math.exp(
        mean_loss
    )


    if was_training:
        model.train()


    return {
        "loss": (
            mean_loss
        ),
        "perplexity": (
            perplexity
        ),
        "active_tokens": (
            total_active_tokens
        ),
    }