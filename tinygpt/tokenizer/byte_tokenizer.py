class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(
        self,
        text: str,
    ) -> list[int]:

        encoded = text.encode(
            "utf-8"
        )

        return list(encoded)

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        byte_sequence = bytes(
            token_ids
        )

        return byte_sequence.decode(
            "utf-8"
        )