from functools import partial

from torch.utils.data import (
    DataLoader,
)

from tinygpt.sft.collate import (
    collate_sft_batch,
)
from tinygpt.sft.dataset import (
    SFTDataset,
)


def create_sft_dataloader(
    dataset: SFTDataset,
    batch_size: int,
    pad_token_id: int,
    shuffle: bool,
) -> DataLoader:

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )


    collate_fn = partial(
        collate_sft_batch,
        pad_token_id=(
            pad_token_id
        ),
    )


    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )