from pathlib import Path

import torch

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
from tinygpt.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
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
from tinygpt.utils.hashing import (
    file_sha256,
)
from tinygpt.utils.random import (
    set_seed,
)



set_seed(42)


device = get_device()


training_config = (
    TrainingConfig()
)


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


model_config = ModelConfig(
    vocab_size=(
        tokenizer.vocab_size
    )
)


tokenizer_hash = file_sha256(
    "data/tokenizer/tokenizer.json"
)


data_hash = file_sha256(
    "data/tokens/metadata.json"
)


dataset = TokenDataset(
    "data/tokens/train.pt"
)


training_generator = (
    torch.Generator()
    .manual_seed(142)
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
    generator=training_generator,
)


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


model.eval()


with torch.no_grad():

    reference_logits = (
        model(x)
        .detach()
        .cpu()
        .clone()
    )
    
    
    
    saved_generator_state = (
    training_generator
    .get_state()
    .clone()
)
    
checkpoint_path = Path(
    "checkpoints/"
    "_roundtrip_test/"
    "checkpoint.pt"
)


save_training_checkpoint(
    path=checkpoint_path,
    model=model,
    optimizer=optimizer,
    completed_step=1,
    best_val_loss=metrics["loss"],
    model_config=model_config,
    training_config=training_config,
    tokenizer_sha256=(
        tokenizer_hash
    ),
    token_data_metadata_sha256=(
        data_hash
    ),
    training_generator=(
        training_generator
    ),
    last_eval={
        "step": 1,
        "train": {
            "loss": metrics["loss"]
        },
        "val": None,
    },
)




restored_model = TinyGPT(
    model_config
).to(device)


restored_optimizer = (
    create_optimizer(
        model=restored_model,
        config=training_config,
    )
)


restored_generator = (
    torch.Generator()
    .manual_seed(999999)
)




checkpoint = (
    load_training_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=(
            restored_optimizer
        ),
        device=device,
        model_config=model_config,
        training_config=(
            training_config
        ),
        tokenizer_sha256=(
            tokenizer_hash
        ),
        token_data_metadata_sha256=(
            data_hash
        ),
        training_generator=(
            restored_generator
        ),
    )
)


restored_model.eval()


with torch.no_grad():

    restored_logits = (
        restored_model(x)
        .detach()
        .cpu()
    )
    
    
    
print("=" * 60)
print("CHECKPOINT ROUND TRIP")
print("=" * 60)


print()
print("Restored step:")
print(
    checkpoint[
        "completed_step"
    ]
)


print()
print(
    "Logits exactly identical:"
)

print(
    torch.equal(
        reference_logits,
        restored_logits,
    )
)


print()
print(
    "Training generator restored:"
)

print(
    torch.equal(
        saved_generator_state,
        restored_generator
        .get_state(),
    )
)


print()
print(
    "Optimizer states before:"
)

print(
    len(
        optimizer.state
    )
)


print()
print(
    "Optimizer states restored:"
)

print(
    len(
        restored_optimizer.state
    )
)