class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))

        self.stoi = {
            char: index
            for index, char in enumerate(chars)
        }

        self.itos = {
            index: char
            for char, index in self.stoi.items()
        }

        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [
            self.stoi[char]
            for char in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.itos[token_id]
            for token_id in token_ids
        )