from dataclasses import dataclass

import torch

from tinygpt.sft.dataset import (
    IGNORE_INDEX,
    SFTExample,
)


@dataclass
class SFTBatch:

    input_ids: torch.Tensor

    targets: torch.Tensor

    attention_mask: torch.Tensor


def collate_sft_batch(
    examples: list[SFTExample],
    pad_token_id: int,
) -> SFTBatch:

    if not examples:
        raise ValueError(
            "Cannot collate an empty batch"
        )


    batch_size = len(
        examples
    )


    max_length = max(
        example.input_ids.numel()
        for example in examples
    )


    input_ids = torch.full(
        size=(
            batch_size,
            max_length,
        ),
        fill_value=(
            pad_token_id
        ),
        dtype=torch.long,
    )


    targets = torch.full(
        size=(
            batch_size,
            max_length,
        ),
        fill_value=(
            IGNORE_INDEX
        ),
        dtype=torch.long,
    )


    attention_mask = torch.zeros(
        size=(
            batch_size,
            max_length,
        ),
        dtype=torch.bool,
    )


    for batch_index, example in enumerate(
        examples
    ):

        length = (
            example.input_ids.numel()
        )


        if (
            example.targets.numel()
            != length
        ):
            raise ValueError(
                "input_ids and targets "
                "must have equal lengths"
            )


        input_ids[
            batch_index,
            :length,
        ] = example.input_ids


        targets[
            batch_index,
            :length,
        ] = example.targets


        attention_mask[
            batch_index,
            :length,
        ] = True


    return SFTBatch(
        input_ids=input_ids,
        targets=targets,
        attention_mask=attention_mask,
    )