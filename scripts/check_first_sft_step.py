import torch

from tinygpt.config import (
    SFTConfig,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.sft.dataset import (
    IGNORE_INDEX,
    SFTDataset,
)
from tinygpt.sft.loss import (
    sft_loss,
)
from tinygpt.sft.optimizer import (
    create_sft_optimizer,
)
from tinygpt.sft.step import (
    sft_train_step,
)
from tinygpt.utils.device import (
    get_device,
)
from tinygpt.utils.random import (
    set_seed,
)


def main():

    config = SFTConfig()


    set_seed(
        config.seed
    )


    device = get_device()


    model, tokenizer, checkpoint = (
        load_model_for_generation(
            checkpoint_path=(
                "checkpoints/"
                "tinystories_5mb_v1/"
                "best.pt"
            ),
            tokenizer_path=(
                "data/tokenizer/"
                "tokenizer.json"
            ),
            device=device,
        )
    )


    print("=" * 70)
    print("FIRST REAL SFT STEP")
    print("=" * 70)


    print()
    print(
        "Loaded base checkpoint step:",
        checkpoint[
            "completed_step"
        ],
    )


    dataset = SFTDataset(
        path=(
            "data/instruction/"
            "train.jsonl"
        ),
        tokenizer=tokenizer,
        context_length=(
            model.config
            .context_length
        ),
    )


    example = dataset[0]


    input_ids = (
        example.input_ids
        .unsqueeze(0)
        .to(device)
    )


    targets = (
        example.targets
        .unsqueeze(0)
        .to(device)
    )


    print()
    print(
        "Input shape:",
        input_ids.shape,
    )


    print(
        "Target shape:",
        targets.shape,
    )


    active_targets = (
        targets
        != IGNORE_INDEX
    )


    print(
        "Assistant target tokens:",
        active_targets
        .sum()
        .item(),
    )


    optimizer = (
        create_sft_optimizer(
            model=model,
            config=config,
        )
    )


    parameter = (
        model
        .transformer
        .blocks[0]
        .attention
        .qkv_projection
        .weight
    )


    before = (
        parameter
        .detach()
        .clone()
    )


    model.eval()


    with torch.no_grad():

        logits_before = model(
            input_ids
        )

        loss_before = sft_loss(
            logits=(
                logits_before
            ),
            targets=targets,
        ).item()


    metrics = sft_train_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        grad_clip_norm=(
            config.grad_clip_norm
        ),
    )


    after = (
        parameter
        .detach()
        .clone()
    )


    changed = (
        not torch.equal(
            before,
            after,
        )
    )


    maximum_change = (
        after
        .sub(before)
        .abs()
        .max()
        .item()
    )


    model.eval()


    with torch.no_grad():

        logits_after = model(
            input_ids
        )

        loss_after = sft_loss(
            logits=(
                logits_after
            ),
            targets=targets,
        ).item()


    print()
    print(
        "Loss before step:",
        round(
            loss_before,
            6,
        ),
    )


    print(
        "Training-step loss:",
        round(
            metrics["loss"],
            6,
        ),
    )


    print(
        "Loss after step:",
        round(
            loss_after,
            6,
        ),
    )


    print()
    print(
        "Gradient norm:",
        round(
            metrics[
                "gradient_norm"
            ],
            6,
        ),
    )


    print()
    print(
        "Pretrained parameter changed:",
        changed,
    )


    print(
        "Maximum absolute change:",
        maximum_change,
    )


if __name__ == "__main__":
    main()