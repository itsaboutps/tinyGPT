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


def main():

    device = get_device()

    training_config = TrainingConfig()


    train_dataset = TokenDataset(
        "data/tokens/train.pt"
    )

    val_dataset = TokenDataset(
        "data/tokens/val.pt"
    )

    test_dataset = TokenDataset(
        "data/tokens/test.pt"
    )


    metrics = evaluate_checkpoint(
        checkpoint_path=(
            "checkpoints/"
            "tinystories_5mb_v1/"
            "best.pt"
        ),
        tokenizer_path=(
            "data/tokenizer/tokenizer.json"
        ),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=(
            training_config.batch_size
        ),
        context_length=128,
        eval_batches=20,
        device=device,
        seed=42,
    )


    print("=" * 70)
    print("BEST CHECKPOINT EVALUATION")
    print("=" * 70)


    print()
    print(
        "Checkpoint step:",
        metrics["checkpoint_step"],
    )


    for split in [
        "train",
        "val",
        "test",
    ]:

        split_metrics = (
            metrics[split]
        )

        print()
        print(
            split.upper()
        )

        print(
            "Loss:",
            round(
                split_metrics["loss"],
                4,
            ),
        )

        print(
            "Perplexity:",
            round(
                split_metrics[
                    "perplexity"
                ],
                2,
            ),
        )


if __name__ == "__main__":
    main()