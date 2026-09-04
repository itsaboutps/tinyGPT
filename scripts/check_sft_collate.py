from tinygpt.config import (
    ModelConfig,
)
from tinygpt.sft.collate import (
    collate_sft_batch,
)
from tinygpt.sft.dataset import (
    SFTDataset,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


def main():

    tokenizer = BPETokenizer.load(
        "data/tokenizer/"
        "tokenizer.json"
    )


    config = ModelConfig(
        vocab_size=(
            tokenizer.vocab_size
        )
    )


    dataset = SFTDataset(
        path=(
            "data/instruction/"
            "train.jsonl"
        ),
        tokenizer=tokenizer,
        context_length=(
            config.context_length
        ),
    )


    examples = [
        dataset[0],
        dataset[1],
        dataset[2],
    ]


    batch = collate_sft_batch(
        examples=examples,
        pad_token_id=(
            tokenizer.eos_token_id
        ),
    )


    print("=" * 70)
    print("SFT COLLATE")
    print("=" * 70)

    print()
    print(
        "Input shape:",
        batch.input_ids.shape,
    )

    print(
        "Target shape:",
        batch.targets.shape,
    )

    print(
        "Mask shape:",
        batch.attention_mask.shape,
    )


    print()
    print(
        "Attention mask:"
    )

    print(
        batch.attention_mask
    )


if __name__ == "__main__":
    main()