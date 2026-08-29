from tinygpt.tokenizer.byte_tokenizer import ByteTokenizer


tokenizer = ByteTokenizer()


examples = [
    "hello",
    "TinyGPT",
    "café",
    "नमस्ते",
    "😀",
]


print("=" * 60)
print("BYTE TOKENIZER")
print("=" * 60)


print()
print("Vocabulary size:")
print(tokenizer.vocab_size)


for text in examples:

    token_ids = tokenizer.encode(
        text
    )

    decoded = tokenizer.decode(
        token_ids
    )

    print()
    print("-" * 40)

    print("Original:")
    print(repr(text))

    print("Token IDs:")
    print(token_ids)

    print("Decoded:")
    print(repr(decoded))

    print("Correct:")
    print(decoded == text)