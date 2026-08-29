from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.utils.device import (
    get_device,
)
from tinygpt.utils.random import (
    set_seed,
)


model_config = ModelConfig()
training_config = TrainingConfig()

device = get_device()


set_seed(
    training_config.seed
)


dataset = TokenDataset(
    "data/tokens/train.pt"
)


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


x, y = dataset.get_batch(
    batch_size=(
        training_config.batch_size
    ),
    context_length=(
        model_config.context_length
    ),
    device=device,
)


print("=" * 60)
print("TINYGPT BATCH")
print("=" * 60)


print()
print("Device:")
print(device)


print()
print("Input shape:")
print(x.shape)


print()
print("Target shape:")
print(y.shape)


print()
print("Input dtype:")
print(x.dtype)


print()
print("Target dtype:")
print(y.dtype)


print()
print("First input:")
print(x[0])


print()
print("First target:")
print(y[0])


print()
print("Decoded first input:")
print(
    repr(
        tokenizer.decode(
            x[0].tolist()
        )
    )
)


print()
print("Decoded first target:")
print(
    repr(
        tokenizer.decode(
            y[0].tolist()
        )
    )
)