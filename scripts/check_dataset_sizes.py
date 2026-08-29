from tinygpt.data.token_dataset import (
    TokenDataset,
)


for split in [
    "train",
    "val",
    "test",
]:

    dataset = TokenDataset(
        f"data/tokens/{split}.pt"
    )

    print(
        f"{split:5s}: "
        f"{len(dataset):,} tokens"
    )