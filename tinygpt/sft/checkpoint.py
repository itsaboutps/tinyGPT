import random

from dataclasses import asdict
from pathlib import Path

import torch

from tinygpt.config import (
    ModelConfig,
    SFTConfig,
)
from tinygpt.training.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    move_optimizer_to_device,
    restore_rng_state,
)


def save_sft_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    completed_epochs: int,
    best_val_loss: float,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    base_checkpoint_path: str,
    base_checkpoint_sha256: str,
    tokenizer_sha256: str,
    train_data_sha256: str,
    val_data_sha256: str,
    last_eval: dict | None,
) -> None:

    checkpoint = {
        "checkpoint_version": 1,

        "phase": "sft",

        "completed_steps": (
            completed_steps
        ),

        "completed_epochs": (
            completed_epochs
        ),

        "best_val_loss": (
            best_val_loss
        ),

        "model_config": (
            asdict(
                model_config
            )
        ),

        "sft_config": (
            asdict(
                sft_config
            )
        ),

        "base_checkpoint_path": (
            base_checkpoint_path
        ),

        "base_checkpoint_sha256": (
            base_checkpoint_sha256
        ),

        "tokenizer_sha256": (
            tokenizer_sha256
        ),

        "train_data_sha256": (
            train_data_sha256
        ),

        "val_data_sha256": (
            val_data_sha256
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "rng_state": (
            capture_rng_state()
        ),

        "last_eval": (
            last_eval
        ),
    }


    atomic_torch_save(
        checkpoint,
        path,
    )
    
    
    
    
    
def validate_sft_checkpoint(
    checkpoint: dict,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    base_checkpoint_sha256: str,
    tokenizer_sha256: str,
    train_data_sha256: str,
    val_data_sha256: str,
) -> None:

    if (
        checkpoint.get(
            "checkpoint_version"
        )
        != 1
    ):
        raise ValueError(
            "Unsupported SFT "
            "checkpoint version"
        )


    if checkpoint.get(
        "phase"
    ) != "sft":

        raise ValueError(
            "Checkpoint is not an "
            "SFT checkpoint"
        )


    if (
        checkpoint["model_config"]
        != asdict(model_config)
    ):
        raise ValueError(
            "Model configuration mismatch"
        )


    if (
        checkpoint["sft_config"]
        != asdict(sft_config)
    ):
        raise ValueError(
            "SFT configuration mismatch"
        )


    if (
        checkpoint[
            "base_checkpoint_sha256"
        ]
        != base_checkpoint_sha256
    ):
        raise ValueError(
            "Base checkpoint mismatch"
        )


    if (
        checkpoint[
            "tokenizer_sha256"
        ]
        != tokenizer_sha256
    ):
        raise ValueError(
            "Tokenizer mismatch"
        )


    if (
        checkpoint[
            "train_data_sha256"
        ]
        != train_data_sha256
    ):
        raise ValueError(
            "SFT training data mismatch"
        )


    if (
        checkpoint[
            "val_data_sha256"
        ]
        != val_data_sha256
    ):
        raise ValueError(
            "SFT validation data mismatch"
        )
        
        
def load_sft_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    base_checkpoint_sha256: str,
    tokenizer_sha256: str,
    train_data_sha256: str,
    val_data_sha256: str,
) -> dict:

    checkpoint_path = Path(
        path
    )


    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: "
            f"{checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    validate_sft_checkpoint(
        checkpoint=checkpoint,
        model_config=model_config,
        sft_config=sft_config,
        base_checkpoint_sha256=(
            base_checkpoint_sha256
        ),
        tokenizer_sha256=(
            tokenizer_sha256
        ),
        train_data_sha256=(
            train_data_sha256
        ),
        val_data_sha256=(
            val_data_sha256
        ),
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    model.to(
        device
    )


    move_optimizer_to_device(
        optimizer,
        device,
    )


    restore_rng_state(
        checkpoint[
            "rng_state"
        ]
    )


    return checkpoint