from pathlib import Path
import hashlib


def load_text(path: str) -> str:
    """
    Load a UTF-8 text file.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Text file does not exist: {file_path}"
        )

    text = file_path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            f"Text file is empty: {file_path}"
        )

    return text


def normalize_text(text: str) -> str:
    """
    Perform minimal, safe normalization.

    We intentionally preserve:
    - uppercase/lowercase
    - punctuation
    - spaces
    - newlines

    because these are meaningful to a language model.
    """

    # Normalize Windows/macOS line endings to "\n".
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null bytes if present.
    text = text.replace("\x00", "")

    return text


def split_text(
    text: str,
    train_fraction: float,
    val_fraction: float,
):
    """
    Split one continuous corpus into
    train, validation, and test regions.
    """

    total_length = len(text)

    train_end = int(
        total_length * train_fraction
    )

    val_end = int(
        total_length
        * (train_fraction + val_fraction)
    )

    train_text = text[:train_end]

    val_text = text[
        train_end:val_end
    ]

    test_text = text[val_end:]

    return train_text, val_text, test_text


def save_text(
    path: str | Path,
    text: str,
):
    """
    Save UTF-8 text, creating parent directories
    when necessary.
    """

    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        text,
        encoding="utf-8",
    )


def text_sha256(text: str) -> str:
    """
    Compute a reproducible fingerprint
    for a text corpus.
    """

    encoded = text.encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()