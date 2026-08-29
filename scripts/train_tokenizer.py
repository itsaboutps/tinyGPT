from tinygpt.config import (
    TokenizerConfig,
)
from tinygpt.data.text import (
    load_text,
)
from tinygpt.tokenizer.bpe_trainer import (
    BPETrainer,
)


config = TokenizerConfig()


print("=" * 60)
print("TINYGPT TOKENIZER TRAINING")
print("=" * 60)


text = load_text(
    "data/processed/train.txt"
)

# ///////////////////////////////////////////////////////////////////////
TOKENIZER_TRAINING_CHARS = (
    500_000
)


tokenizer_training_text = (
    text[
        :TOKENIZER_TRAINING_CHARS
    ]
)


print(
    "Full training characters:",
    len(text),
)


print(
    "Tokenizer training characters:",
    len(
        tokenizer_training_text
    ),
)


trainer = BPETrainer(
    target_vocab_size=(
        config.vocab_size
    ),
    min_pair_frequency=(
        config
        .min_pair_frequency
    ),
)


tokenizer = trainer.train(
    tokenizer_training_text
)

tokenizer.save(
    config.output_path
)


sample = text[:5000]


encoded = tokenizer.encode(
    sample
)


sample_bytes = len(
    sample.encode("utf-8")
)


token_count = len(
    encoded
)


bytes_per_token = (
    sample_bytes
    / token_count
)


print()
print("=" * 60)
print("TOKENIZER SUMMARY")
print("=" * 60)


print()
print("Vocabulary size:")
print(
    tokenizer.vocab_size
)


print()
print("Sample bytes:")
print(
    sample_bytes
)


print()
print("Sample BPE tokens:")
print(
    token_count
)


print()
print("Bytes per token:")
print(
    round(
        bytes_per_token,
        3,
    )
)


print()
print("Round trip:")
print(
    tokenizer.decode(encoded)
    == sample
)


print()
print("Saved tokenizer:")
print(
    config.output_path
)