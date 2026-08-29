from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.model.gpt import (
    TinyGPT,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.training.optimizer import (
    create_optimizer,
)
from tinygpt.training.step import (
    train_step,
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
    vocab_size=(
        tokenizer.vocab_size
    )
)


dataset = TokenDataset(
    "data/tokens/train.pt"
)


model = TinyGPT(
    model_config
).to(device)


optimizer = create_optimizer(
    model=model,
    config=training_config,
)


x, y = dataset.get_batch(
    batch_size=2,
    context_length=32,
    device=device,
)


for step in range(101):

    metrics = train_step(
        model=model,
        optimizer=optimizer,
        x=x,
        y=y,
        grad_clip_norm=(
            training_config
            .grad_clip_norm
        ),
    )

    if (
        step == 0
        or step % 10 == 0
    ):

        print(
            f"Step {step:3d} | "
            f"loss "
            f"{metrics['loss']:.4f} | "
            f"grad norm "
            f"{metrics['gradient_norm']:.4f}"
        )