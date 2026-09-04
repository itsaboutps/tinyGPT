from tinygpt.config import (
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.training.evaluate_checkpoint import (
    evaluate_checkpoint,
)
from tinygpt.utils.device import (
    get_device,
)


def print_metrics(
    name: str,
    metrics: dict,
):

    print()
    print(name)
    print("-" * 70)

    print(
        "Checkpoint step:",
        metrics[
            "checkpoint_step"
        ],
    )

    for split in [
        "train",
        "val",
        "test",
    ]:

        print(
            f"{split:5s} | "
            f"loss "
            f"{metrics[split]['loss']:.4f} | "
            f"ppl "
            f"{metrics[split]['perplexity']:.2f}"
        )


def main():

    device = get_device()

    config = TrainingConfig()


    train_dataset = TokenDataset(
        "data/tokens/train.pt"
    )

    val_dataset = TokenDataset(
        "data/tokens/val.pt"
    )

    test_dataset = TokenDataset(
        "data/tokens/test.pt"
    )


    common = {
        "tokenizer_path": (
            "data/tokenizer/tokenizer.json"
        ),
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "batch_size": (
            config.batch_size
        ),
        "context_length": 128,
        "eval_batches": 20,
        "device": device,
        "seed": 42,
    }


    best = evaluate_checkpoint(
        checkpoint_path=(
            "checkpoints/"
            "tinystories_5mb_v1/"
            "best.pt"
        ),
        **common,
    )


    latest = evaluate_checkpoint(
        checkpoint_path=(
            "checkpoints/"
            "tinystories_5mb_v1/"
            "latest.pt"
        ),
        **common,
    )


    print("=" * 70)
    print("CHECKPOINT COMPARISON")
    print("=" * 70)

    print_metrics(
        "BEST",
        best,
    )

    print_metrics(
        "LATEST",
        latest,
    )


if __name__ == "__main__":
    main()