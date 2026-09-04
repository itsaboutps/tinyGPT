from tinygpt.config import (
    SFTConfig,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.sft.dataloader import (
    create_sft_dataloader,
)
from tinygpt.sft.dataset import (
    SFTDataset,
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


def main():

    config = SFTConfig()


    device = get_device()


    model, tokenizer, _ = (
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


    loader = create_sft_dataloader(
        dataset=dataset,
        batch_size=4,
        pad_token_id=(
            tokenizer.eos_token_id
        ),
        shuffle=False,
    )


    batch = next(
        iter(loader)
    )


    print("=" * 70)
    print("BATCHED SFT STEP")
    print("=" * 70)


    print()
    print(
        "Input:",
        batch.input_ids.shape,
    )


    print(
        "Targets:",
        batch.targets.shape,
    )


    print(
        "Attention mask:",
        batch.attention_mask.shape,
    )


    optimizer = (
        create_sft_optimizer(
            model=model,
            config=config,
        )
    )


    metrics = sft_train_step(
        model=model,
        optimizer=optimizer,
        input_ids=(
            batch.input_ids
            .to(device)
        ),
        targets=(
            batch.targets
            .to(device)
        ),
        attention_mask=(
            batch.attention_mask
            .to(device)
        ),
        grad_clip_norm=(
            config.grad_clip_norm
        ),
    )


    print()
    print(
        "Loss:",
        metrics["loss"],
    )


    print(
        "Gradient norm:",
        metrics[
            "gradient_norm"
        ],
    )


if __name__ == "__main__":
    main()