from tinygpt.config import (
    ModelConfig,
)
from tinygpt.sft.dataset import (
    IGNORE_INDEX,
    SFTDataset,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


def main():

    tokenizer = (
        BPETokenizer.load(
            "data/tokenizer/"
            "tokenizer.json"
        )
    )


    model_config = (
        ModelConfig(
            vocab_size=(
                tokenizer.vocab_size
            )
        )
    )


    dataset = SFTDataset(
        path=(
            "data/instruction/"
            "train.jsonl"
        ),
        tokenizer=tokenizer,
        context_length=(
            model_config
            .context_length
        ),
    )


    print("=" * 70)
    print("SFT DATASET")
    print("=" * 70)


    print()
    print(
        "Examples:",
        len(dataset),
    )


    example = dataset[0]


    print()
    print(
        "Input shape:",
        example.input_ids.shape,
    )

    print(
        "Target shape:",
        example.targets.shape,
    )


    ignored = (
        example.targets
        == IGNORE_INDEX
    )


    active = (
        example.targets
        != IGNORE_INDEX
    )


    print()
    print(
        "Ignored targets:",
        ignored.sum().item(),
    )


    print(
        "Assistant targets:",
        active.sum().item(),
    )


    active_ids = (
        example.targets[
            active
        ]
        .tolist()
    )


    print()
    print(
        "Decoded active targets:"
    )


    print(
        tokenizer.decode(
            active_ids,
            skip_special_tokens=True,
        )
    )


if __name__ == "__main__":
    main()