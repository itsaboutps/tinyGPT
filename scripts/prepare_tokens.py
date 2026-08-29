import hashlib
import json
from pathlib import Path

import torch

from tinygpt.config import TokenizerConfig
from tinygpt.data.text import load_text
from tinygpt.tokenizer.bpe import BPETokenizer


tokenizer_config = TokenizerConfig()


TOKENIZER_PATH = Path(
    tokenizer_config.output_path
)

PROCESSED_DIR = Path(
    "data/processed"
)

TOKEN_OUTPUT_DIR = Path(
    "data/tokens"
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def encode_split(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
):
    text = load_text(
        str(input_path)
    )

    # token_ids = tokenizer.encode(
    #     text
    # )
    token_ids = encode_documents(
                                tokenizer,
                                text,
                                )

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    torch.save(
        tokens,
        output_path,
    )

    return {
        "characters": len(text),
        "utf8_bytes": len(
            text.encode("utf-8")
        ),
        "tokens": len(token_ids),
    }


def main():
    print("=" * 60)
    print("TINYGPT TOKEN PREPARATION")
    print("=" * 60)

    tokenizer = BPETokenizer.load(
        str(TOKENIZER_PATH)
    )

    TOKEN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "tokenizer_path": str(
            TOKENIZER_PATH
        ),
        "tokenizer_sha256": (
            sha256_file(
                TOKENIZER_PATH
            )
        ),
        "vocab_size": (
            tokenizer.vocab_size
        ),
        "splits": {},
    }

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        input_path = (
            PROCESSED_DIR
            / f"{split_name}.txt"
        )

        output_path = (
            TOKEN_OUTPUT_DIR
            / f"{split_name}.pt"
        )

        print()
        print(
            f"Encoding {split_name}..."
        )

        stats = encode_split(
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_path,
        )

        metadata["splits"][
            split_name
        ] = stats

        print(
            "Characters:",
            stats["characters"],
        )

        print(
            "UTF-8 bytes:",
            stats["utf8_bytes"],
        )

        print(
            "Tokens:",
            stats["tokens"],
        )

    metadata_path = (
        TOKEN_OUTPUT_DIR
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Tokenized datasets saved to:")
    print(TOKEN_OUTPUT_DIR)
    
    
def encode_documents(
    tokenizer,
    text: str,
) -> list[int]:

    documents = text.split(
        "\n\n"
    )

    all_token_ids = []


    for document in documents:

        document = (
            document.strip()
        )

        if not document:
            continue


        token_ids = tokenizer.encode(
            document,
            add_eos=True,
        )


        all_token_ids.extend(
            token_ids
        )


    return all_token_ids


if __name__ == "__main__":
    main()