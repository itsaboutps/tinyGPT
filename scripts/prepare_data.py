import json
from pathlib import Path

from tinygpt.config import DataConfig
from tinygpt.data.text import (
    load_text,
    normalize_text,
    save_text,
    split_text,
    text_sha256,
)


config = DataConfig()


print("=" * 60)
print("TINYGPT DATA PREPARATION")
print("=" * 60)


# ---------------------------------------------------------
# Load raw data
# ---------------------------------------------------------

raw_text = load_text(
    config.raw_path
)

print()
print("Raw characters:")
print(len(raw_text))


# ---------------------------------------------------------
# Normalize
# ---------------------------------------------------------

text = normalize_text(
    raw_text
)

print()
print("Normalized characters:")
print(len(text))


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------

train_text, val_text, test_text = split_text(
    text=text,
    train_fraction=config.train_fraction,
    val_fraction=config.val_fraction,
)


# ---------------------------------------------------------
# Validate split
# ---------------------------------------------------------

assert (
    len(train_text)
    + len(val_text)
    + len(test_text)
    == len(text)
)


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------

output_dir = Path(
    config.processed_dir
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------

save_text(
    output_dir / "train.txt",
    train_text,
)

save_text(
    output_dir / "val.txt",
    val_text,
)

save_text(
    output_dir / "test.txt",
    test_text,
)


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

metadata = {
    "source": config.raw_path,

    "sha256": text_sha256(text),

    "total_characters": len(text),

    "train_characters": len(train_text),

    "validation_characters": len(val_text),

    "test_characters": len(test_text),

    "train_fraction": config.train_fraction,

    "validation_fraction": config.val_fraction,

    "test_fraction": config.test_fraction,
}


metadata_path = (
    output_dir / "metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()

print("Train characters:")
print(len(train_text))

print()

print("Validation characters:")
print(len(val_text))

print()

print("Test characters:")
print(len(test_text))

print()

print("Corpus SHA-256:")
print(metadata["sha256"])

print()

print("Processed data saved to:")
print(output_dir)