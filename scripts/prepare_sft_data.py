import json
from pathlib import Path


SYSTEM = (
    "You are a helpful and concise assistant."
)


TRAIN_EXAMPLES = [
    {
        "system": SYSTEM,
        "user": "What is 2 + 2?",
        "assistant": "2 + 2 equals 4.",
    },
    {
        "system": SYSTEM,
        "user": "What color is grass?",
        "assistant": "Grass is usually green.",
    },
    {
        "system": SYSTEM,
        "user": "Say hello politely.",
        "assistant": "Hello! It is nice to meet you.",
    },
    {
        "system": SYSTEM,
        "user": "What is the opposite of hot?",
        "assistant": "The opposite of hot is cold.",
    },
    {
        "system": SYSTEM,
        "user": "What is a cat?",
        "assistant": (
            "A cat is a small animal "
            "that is often kept as a pet."
        ),
    },
    {
        "system": SYSTEM,
        "user": "Name one fruit.",
        "assistant": "An apple is a fruit.",
    },
    {
        "system": SYSTEM,
        "user": "What comes after Monday?",
        "assistant": "Tuesday comes after Monday.",
    },
    {
        "system": SYSTEM,
        "user": "Finish this sentence: The sky is",
        "assistant": "The sky is blue.",
    },
]


VAL_EXAMPLES = [
    {
        "system": SYSTEM,
        "user": "What is 3 + 3?",
        "assistant": "3 + 3 equals 6.",
    },
    {
        "system": SYSTEM,
        "user": "What color is snow?",
        "assistant": "Snow is usually white.",
    },
]


TEST_EXAMPLES = [
    {
        "system": SYSTEM,
        "user": "What is 1 + 1?",
        "assistant": "1 + 1 equals 2.",
    },
    {
        "system": SYSTEM,
        "user": "Name one animal.",
        "assistant": "A dog is an animal.",
    },
]


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            line = json.dumps(
                record,
                ensure_ascii=False,
            )

            file.write(
                line + "\n"
            )


def main():

    output_dir = Path(
        "data/instruction"
    )


    write_jsonl(
        output_dir / "train.jsonl",
        TRAIN_EXAMPLES,
    )

    write_jsonl(
        output_dir / "val.jsonl",
        VAL_EXAMPLES,
    )

    write_jsonl(
        output_dir / "test.jsonl",
        TEST_EXAMPLES,
    )


    print("=" * 60)
    print("SFT DATA PREPARED")
    print("=" * 60)

    print(
        "Train examples:",
        len(TRAIN_EXAMPLES),
    )

    print(
        "Validation examples:",
        len(VAL_EXAMPLES),
    )

    print(
        "Test examples:",
        len(TEST_EXAMPLES),
    )


if __name__ == "__main__":
    main()