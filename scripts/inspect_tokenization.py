from tinygpt.data.text import load_text
from tinygpt.tokenizer.byte_tokenizer import ByteTokenizer


text = load_text(
    "data/processed/train.txt"
)


tokenizer = ByteTokenizer()


sample = text[:200]


token_ids = tokenizer.encode(
    sample
)


print("=" * 60)
print("TOKENIZATION INSPECTION")
print("=" * 60)


print()
print("Text:")
print(repr(sample))


print()
print("Characters:")
print(len(sample))


print()
print("Tokens:")
print(len(token_ids))


print()
print("First 100 token IDs:")
print(token_ids[:100])


print()
print("Round trip correct:")
print(
    tokenizer.decode(token_ids)
    == sample
)