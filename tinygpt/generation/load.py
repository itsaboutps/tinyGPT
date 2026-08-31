from pathlib import Path

import torch

from tinygpt.config import (
    ModelConfig,
)
from tinygpt.model.gpt import (
    TinyGPT,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.utils.hashing import (
    file_sha256,
)


def load_model_for_generation(
    checkpoint_path: str,
    tokenizer_path: str,
    device: torch.device,
):

    checkpoint_file = Path(
        checkpoint_path
    )

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_file}"
        )


    tokenizer = BPETokenizer.load(
        tokenizer_path
    )


    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )


    if (
        "model_config"
        not in checkpoint
    ):
        raise ValueError(
            "Checkpoint does not contain "
            "model_config"
        )


    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise ValueError(
            "Checkpoint does not contain "
            "model_state_dict"
        )


    current_tokenizer_hash = (
        file_sha256(
            tokenizer_path
        )
    )


    saved_tokenizer_hash = (
        checkpoint.get(
            "tokenizer_sha256"
        )
    )


    if (
        saved_tokenizer_hash
        != current_tokenizer_hash
    ):
        raise ValueError(
            "Tokenizer does not match "
            "checkpoint"
        )


    model_config = ModelConfig(
        **checkpoint[
            "model_config"
        ]
    )


    if (
        tokenizer.vocab_size
        != model_config.vocab_size
    ):
        raise ValueError(
            "Tokenizer vocabulary size "
            "does not match model"
        )


    model = TinyGPT(
        model_config
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        device
    )


    model.eval()


    return (
        model,
        tokenizer,
        checkpoint,
    )