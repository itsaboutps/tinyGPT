from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path(
    "data/raw/input.txt"
)

TARGET_BYTES = (
    5 * 1024 * 1024
)


def main():

    print("=" * 60)
    print("TINYSTORIES SUBSET DOWNLOAD")
    print("=" * 60)

    print()
    print(
        "Target size:",
        TARGET_BYTES,
        "bytes",
    )


    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    bytes_written = 0
    story_count = 0


    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for example in dataset:

            story = example["text"]

            if not story:
                continue


            document = (
                story.strip()
                + "\n\n"
            )


            encoded_size = len(
                document.encode(
                    "utf-8"
                )
            )


            if (
                bytes_written
                + encoded_size
                > TARGET_BYTES
            ):
                break


            file.write(
                document
            )


            bytes_written += (
                encoded_size
            )

            story_count += 1


            if story_count % 1000 == 0:

                print(
                    f"Stories: "
                    f"{story_count:,} | "
                    f"MB: "
                    f"{bytes_written / (1024 ** 2):.2f}"
                )


    print()
    print("=" * 60)

    print(
        "Stories written:",
        story_count,
    )

    print(
        "Final size MB:",
        round(
            bytes_written
            / (1024 ** 2),
            2,
        ),
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()