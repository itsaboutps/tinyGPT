from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.model.gpt import TinyGPT
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.training.loss import (
    language_model_loss,
)
from tinygpt.utils.device import (
    get_device,
)
from tinygpt.utils.random import (
    set_seed,
)

import math


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


model = TinyGPT(
    model_config
).to(device)


logits = model(
    x
)


loss = language_model_loss(
    logits=logits,
    targets=y,
)


print("=" * 60)
print("TINYGPT LANGUAGE MODEL LOSS")
print("=" * 60)


print()
print("Input:")
print(x.shape)


print()
print("Targets:")
print(y.shape)


print()
print("Logits:")
print(logits.shape)


print()
print("Loss:")
print(loss)


print()
print("Loss value:")
print(loss.item())



random_baseline = math.log(
    model_config.vocab_size
)


print()
print(
    "Uniform random baseline:"
)

print(
    random_baseline
)