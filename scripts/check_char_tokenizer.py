from tinygpt.config import DataConfig
from tinygpt.data.text import load_text
from tinygpt.tokenizer.char_tokenizer import CharTokenizer


config = DataConfig()


text = load_text(
    "data/processed/train.txt"
)


tokenizer = CharTokenizer(text)


print("=" * 60)
print("CHARACTER TOKENIZER")
print("=" * 60)


print()
print("Vocabulary size:")
print(tokenizer.vocab_size)


print()
print("First 20 vocabulary entries:")

for token_id in range(
    min(20, tokenizer.vocab_size)
):
    char = tokenizer.itos[token_id]

    print(
        token_id,
        repr(char)
    )


example = "the"


print()
print("Original:")
print(repr(example))


encoded = tokenizer.encode(
    example
)


print()
print("Encoded:")
print(encoded)


decoded = tokenizer.decode(
    encoded
)


print()
print("Decoded:")
print(repr(decoded))


print()
print("Round trip correct:")
print(decoded == example)