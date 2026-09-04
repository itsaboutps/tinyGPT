from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.model.gpt import (
    TinyGPT,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.training.evaluate import (
    evaluate_loss,
)
from tinygpt.utils.device import (
    get_device,
)
from tinygpt.utils.random import (
    set_seed,
)


def main():

    training_config = (
        TrainingConfig()
    )

    set_seed(
        training_config.seed
    )

    device = get_device()


    tokenizer = BPETokenizer.load(
        "data/tokenizer/tokenizer.json"
    )


    model_config = ModelConfig(
        vocab_size=(
            tokenizer.vocab_size
        )
    )


    model = TinyGPT(
        model_config
    ).to(device)


    datasets = {
        "train": TokenDataset(
            "data/tokens/train.pt"
        ),
        "val": TokenDataset(
            "data/tokens/val.pt"
        ),
        "test": TokenDataset(
            "data/tokens/test.pt"
        ),
    }


    print("=" * 70)
    print("UNTRAINED MODEL")
    print("=" * 70)


    for index, (
        split,
        dataset,
    ) in enumerate(
        datasets.items()
    ):

        metrics = evaluate_loss(
            model=model,
            dataset=dataset,
            batch_size=(
                training_config
                .batch_size
            ),
            context_length=(
                model_config
                .context_length
            ),
            num_batches=20,
            device=device,
            seed=(
                42
                + index * 100
            ),
        )


        print(
            f"{split:5s} | "
            f"loss "
            f"{metrics['loss']:.4f} | "
            f"ppl "
            f"{metrics['perplexity']:.2f}"
        )


if __name__ == "__main__":
    main()