import torch
import torch.nn.functional as F


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape "
            "[B, T, V]"
        )

    if targets.ndim != 2:
        raise ValueError(
            "targets must have shape "
            "[B, T]"
        )

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "Batch and sequence dimensions "
            "must match between logits "
            "and targets"
        )

    if targets.dtype != torch.long:
        raise ValueError(
            "targets must use "
            "torch.long dtype"
        )


    vocab_size = logits.shape[-1]


    logits_flat = logits.reshape(
        -1,
        vocab_size,
    )


    targets_flat = targets.reshape(
        -1
    )


    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
    )


    return loss