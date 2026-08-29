from tinygpt.config import (
    ModelConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


model_config = ModelConfig()


dataset = TokenDataset(
    "data/tokens/train.pt"
)


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


print("=" * 60)
print("TOKEN DATASET")
print("=" * 60)


print()
print("Dataset tokens:")
print(len(dataset))


x, y = dataset.get_window(
    start=0,
    context_length=(
        min(
            model_config.context_length,
            32,
        )
    ),
)


print()
print("Input shape:")
print(x.shape)


print()
print("Target shape:")
print(y.shape)


print()
print("Input IDs:")
print(x)


print()
print("Target IDs:")
print(y)


print()
print("Decoded input:")
print(
    repr(
        tokenizer.decode(
            x.tolist()
        )
    )
)


print()
print("Decoded target:")
print(
    repr(
        tokenizer.decode(
            y.tolist()
        )
    )
)