import torch

from tinygpt.data.token_dataset import (
    TokenDataset,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.training.evaluate import (
    evaluate_loss,
)


def evaluate_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    train_dataset: TokenDataset,
    val_dataset: TokenDataset,
    test_dataset: TokenDataset,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
) -> dict:

    model, tokenizer, checkpoint = (
        load_model_for_generation(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )
    )


    train_metrics = evaluate_loss(
        model=model,
        dataset=train_dataset,
        batch_size=batch_size,
        context_length=context_length,
        num_batches=eval_batches,
        device=device,
        seed=seed + 100,
    )


    val_metrics = evaluate_loss(
        model=model,
        dataset=val_dataset,
        batch_size=batch_size,
        context_length=context_length,
        num_batches=eval_batches,
        device=device,
        seed=seed + 200,
    )


    test_metrics = evaluate_loss(
        model=model,
        dataset=test_dataset,
        batch_size=batch_size,
        context_length=context_length,
        num_batches=eval_batches,
        device=device,
        seed=seed + 300,
    )


    return {
        "checkpoint_step": (
            checkpoint["completed_step"]
        ),
        "best_val_loss": (
            checkpoint["best_val_loss"]
        ),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }