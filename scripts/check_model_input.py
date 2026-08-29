from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.model.embeddings import (
    TokenEmbedding,
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


training_config = TrainingConfig()


set_seed(
    training_config.seed
)


device = get_device()


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


model_config = ModelConfig(
    vocab_size=tokenizer.vocab_size
)


dataset = TokenDataset(
    "data/tokens/train.pt"
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


embedding = TokenEmbedding(
    vocab_size=(
        model_config.vocab_size
    ),
    d_model=(
        model_config.d_model
    ),
).to(device)


token_embeddings = embedding(
    x
)


print("=" * 60)
print("TINYGPT MODEL INPUT")
print("=" * 60)


print()
print("Input token IDs:")
print(x.shape)


print()
print("Targets:")
print(y.shape)


print()
print("Token embeddings:")
print(
    token_embeddings.shape
)


print()
print("Expected:")
print(
    (
        training_config.batch_size,
        model_config.context_length,
        model_config.d_model,
    )
)