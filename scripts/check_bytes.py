examples = [
    "A",
    "hello",
    "café",
    "😀",
]


for text in examples:

    encoded = text.encode("utf-8")

    byte_values = list(encoded)

    print("=" * 50)

    print("Text:")
    print(repr(text))

    print()

    print("UTF-8 bytes:")
    print(encoded)

    print()

    print("Byte values:")
    print(byte_values)

    print()

    print("Recovered:")
    print(
        encoded.decode("utf-8")
    )