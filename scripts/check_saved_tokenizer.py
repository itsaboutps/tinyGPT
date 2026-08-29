from tinygpt.config import (
    TokenizerConfig,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer
)


config = TokenizerConfig()


tokenizer = BPETokenizer.load(
    config.output_path
)


examples = [
    "hello world",
    "TinyGPT is learning.",
    "café",
    "नमस्ते",
    "😀",
]


print("=" * 60)
print("SAVED TOKENIZER TEST")
print("=" * 60)


print()
print("Vocabulary size:")
print(
    tokenizer.vocab_size
)


for text in examples:

    token_ids = tokenizer.encode(
        text
    )

    decoded = tokenizer.decode(
        token_ids
    )

    print()
    print("-" * 50)

    print("Text:")
    print(repr(text))

    print("Token IDs:")
    print(token_ids)

    print("Decoded:")
    print(repr(decoded))

    print("Correct:")
    print(
        decoded == text
    )
    token_ids = tokenizer.encode(
    "hello",
    add_eos=True,
    )

    print(token_ids)

    print(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )
    )