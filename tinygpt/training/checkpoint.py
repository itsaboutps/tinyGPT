import json
import os
import platform
import random

from dataclasses import asdict
from pathlib import Path

import torch

from tinygpt.config import (
    ModelConfig,
    RunConfig,
    TrainingConfig,
)



def capture_rng_state(
    training_generator: (
        torch.Generator | None
    ) = None,
) -> dict:

    state = {
        "python_random": (
            random.getstate()
        ),
        "torch": (
            torch.get_rng_state()
        ),
    }


    if torch.cuda.is_available():

        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    else:

        state["cuda"] = None


    if training_generator is not None:

        state["training_generator"] = (
            training_generator
            .get_state()
        )

    else:

        state[
            "training_generator"
        ] = None


    return state




def restore_rng_state(
    state: dict,
    training_generator: (
        torch.Generator | None
    ) = None,
) -> None:

    random.setstate(
        state["python_random"]
    )


    torch.set_rng_state(
        state["torch"].cpu()
    )


    cuda_state = state.get(
        "cuda"
    )

    if (
        torch.cuda.is_available()
        and cuda_state is not None
    ):

        torch.cuda.set_rng_state_all(
            [
                item.cpu()
                for item in cuda_state
            ]
        )


    generator_state = state.get(
        "training_generator"
    )

    if (
        training_generator
        is not None
        and generator_state
        is not None
    ):

        training_generator.set_state(
            generator_state.cpu()
        )
        
        
        
def atomic_torch_save(
    payload: dict,
    path: str | Path,
) -> None:

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temp_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )


    torch.save(
        payload,
        temp_path,
    )


    os.replace(
        temp_path,
        output_path,
    )
    
    
    
def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_step: int,
    best_val_loss: float,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer_sha256: str,
    token_data_metadata_sha256: str,
    training_generator: (
        torch.Generator | None
    ),
    last_eval: dict | None,
) -> None:

    checkpoint = {
        "checkpoint_version": 1,

        "completed_step": (
            completed_step
        ),

        "best_val_loss": (
            best_val_loss
        ),

        "model_config": (
            asdict(
                model_config
            )
        ),

        "training_config": (
            asdict(
                training_config
            )
        ),

        "tokenizer_sha256": (
            tokenizer_sha256
        ),

        "token_data_metadata_sha256": (
            token_data_metadata_sha256
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "rng_state": (
            capture_rng_state(
                training_generator
            )
        ),

        "last_eval": (
            last_eval
        ),
    }


    atomic_torch_save(
        checkpoint,
        path,
    )
    
    
    
def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:

    for state in optimizer.state.values():

        for key, value in state.items():

            if isinstance(
                value,
                torch.Tensor,
            ):

                state[key] = value.to(
                    device
                )
                
                
def validate_checkpoint(
    checkpoint: dict,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer_sha256: str,
    token_data_metadata_sha256: str,
) -> None:

    if (
        checkpoint.get(
            "checkpoint_version"
        )
        != 1
    ):
        raise ValueError(
            "Unsupported checkpoint version"
        )


    if (
        checkpoint["model_config"]
        != asdict(model_config)
    ):
        raise ValueError(
            "Model configuration does not "
            "match checkpoint"
        )


    if (
        checkpoint[
            "training_config"
        ]
        != asdict(training_config)
    ):
        raise ValueError(
            "Training configuration does not "
            "match checkpoint"
        )


    if (
        checkpoint[
            "tokenizer_sha256"
        ]
        != tokenizer_sha256
    ):
        raise ValueError(
            "Tokenizer does not match "
            "checkpoint"
        )


    if (
        checkpoint[
            "token_data_metadata_sha256"
        ]
        != token_data_metadata_sha256
    ):
        raise ValueError(
            "Tokenized dataset metadata "
            "does not match checkpoint"
        )
        
        
def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer_sha256: str,
    token_data_metadata_sha256: str,
    training_generator: (
        torch.Generator | None
    ),
) -> dict:

    checkpoint_path = Path(
        path
    )


    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    validate_checkpoint(
        checkpoint=checkpoint,
        model_config=model_config,
        training_config=(
            training_config
        ),
        tokenizer_sha256=(
            tokenizer_sha256
        ),
        token_data_metadata_sha256=(
            token_data_metadata_sha256
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
        checkpoint["rng_state"],
        training_generator,
    )


    return checkpoint


def write_run_metadata(
    path: str | Path,
    run_config: RunConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer_sha256: str,
    token_data_metadata_sha256: str,
) -> None:

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    metadata = {
        "experiment_name": (
            run_config.experiment_name
        ),

        "model_config": (
            asdict(model_config)
        ),

        "training_config": (
            asdict(training_config)
        ),

        "tokenizer_sha256": (
            tokenizer_sha256
        ),

        "token_data_metadata_sha256": (
            token_data_metadata_sha256
        ),

        "python_version": (
            platform.python_version()
        ),

        "torch_version": (
            torch.__version__
        ),
    }


    output_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )