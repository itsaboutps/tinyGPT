import os
import random

from dataclasses import asdict
from pathlib import Path

import torch

from tinygpt.config import (
    SFTConfig,
    SFTRunConfig,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.sft.dataset import (
    SFTDataset,
)
from tinygpt.sft.evaluate import (
    evaluate_sft_loss,
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
from tinygpt.utils.hashing import (
    file_sha256,
)
from tinygpt.utils.random import (
    set_seed,
)
from tinygpt.sft.dataloader import (
    create_sft_dataloader,
)
from tinygpt.sft.dataloader import (
    create_sft_dataloader,
)

TOKENIZER_PATH = (
    "data/tokenizer/tokenizer.json"
)

TRAIN_PATH = (
    "data/instruction/train.jsonl"
)

VAL_PATH = (
    "data/instruction/val.jsonl"
)


def save_chat_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sft_config: SFTConfig,
    completed_steps: int,
    completed_epochs: int,
    best_val_loss: float,
    tokenizer_hash: str,
    base_checkpoint_path: str,
    base_checkpoint_hash: str,
    last_eval: dict,
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
                model.config
            )
        ),

        "sft_config": (
            asdict(
                sft_config
            )
        ),

        "tokenizer_sha256": (
            tokenizer_hash
        ),

        "base_checkpoint_path": (
            base_checkpoint_path
        ),

        "base_checkpoint_sha256": (
            base_checkpoint_hash
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "last_eval": (
            last_eval
        ),
    }


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temporary_path = Path(
        str(path) + ".tmp"
    )


    torch.save(
        checkpoint,
        temporary_path,
    )


    os.replace(
        temporary_path,
        path,
    )


def main():

    sft_config = (
        SFTConfig()
    )

    run_config = (
        SFTRunConfig()
    )


    set_seed(
        sft_config.seed
    )


    device = get_device()


    run_dir = (
        Path(
            run_config
            .checkpoint_root
        )
        /
        run_config
        .experiment_name
    )


    latest_path = (
        run_dir
        / "latest_chat.pt"
    )


    best_path = (
        run_dir
        / "best_chat.pt"
    )


    tokenizer_hash = (
        file_sha256(
            TOKENIZER_PATH
        )
    )


    base_checkpoint_hash = (
        file_sha256(
            run_config
            .base_checkpoint
        )
    )


    model, tokenizer, base_checkpoint = (
        load_model_for_generation(
            checkpoint_path=(
                run_config
                .base_checkpoint
            ),
            tokenizer_path=(
                TOKENIZER_PATH
            ),
            device=device,
        )
    )


    train_dataset = SFTDataset(
        path=TRAIN_PATH,
        tokenizer=tokenizer,
        context_length=(
            model.config
            .context_length
        ),
    )
    train_loader = (
        create_sft_dataloader(
            dataset=train_dataset,
            batch_size=(
                sft_config.batch_size
            ),
            pad_token_id=(
                tokenizer.eos_token_id
            ),
            shuffle=True,
        )
    )

    val_dataset = SFTDataset(
        path=VAL_PATH,
        tokenizer=tokenizer,
        context_length=(
            model.config
            .context_length
        ),
    )


    optimizer = (
        create_sft_optimizer(
            model=model,
            config=sft_config,
        )
    )


    print("=" * 70)
    print("TINYCHATGPT SFT")
    print("=" * 70)

    print()
    print(
        "Device:",
        device,
    )

    print(
        "Base checkpoint:",
        run_config.base_checkpoint,
    )

    print(
        "Base checkpoint step:",
        base_checkpoint[
            "completed_step"
        ],
    )

    print(
        "Train examples:",
        len(train_dataset),
    )

    print(
        "Validation examples:",
        len(val_dataset),
    )


    initial_val = (
        evaluate_sft_loss(
            model=model,
    dataset=val_dataset,
    tokenizer=tokenizer,
    batch_size=(
        sft_config.batch_size
    ),
    device=device,
        )
    )


    best_val_loss = (
        initial_val["loss"]
    )


    completed_steps = 0

    completed_epochs = 0


    last_eval = {
        "step": 0,
        "epoch": 0,
        "val": (
            initial_val
        ),
    }


    print()
    print("=" * 70)
    print("INITIAL SFT EVALUATION")
    print("=" * 70)

    print(
        f"Validation loss "
        f"{initial_val['loss']:.4f} | "
        f"ppl "
        f"{initial_val['perplexity']:.2f}"
    )


    #
    # Save step-zero checkpoints.
    #
    # This guarantees that the SFT run directory
    # is valid before training begins.
    #

    save_chat_checkpoint(
        path=latest_path,
        model=model,
        optimizer=optimizer,
        sft_config=sft_config,
        completed_steps=(
            completed_steps
        ),
        completed_epochs=(
            completed_epochs
        ),
        best_val_loss=(
            best_val_loss
        ),
        tokenizer_hash=(
            tokenizer_hash
        ),
        base_checkpoint_path=(
            run_config
            .base_checkpoint
        ),
        base_checkpoint_hash=(
            base_checkpoint_hash
        ),
        last_eval=(
            last_eval
        ),
    )


    save_chat_checkpoint(
        path=best_path,
        model=model,
        optimizer=optimizer,
        sft_config=sft_config,
        completed_steps=(
            completed_steps
        ),
        completed_epochs=(
            completed_epochs
        ),
        best_val_loss=(
            best_val_loss
        ),
        tokenizer_hash=(
            tokenizer_hash
        ),
        base_checkpoint_path=(
            run_config
            .base_checkpoint
        ),
        base_checkpoint_hash=(
            base_checkpoint_hash
        ),
        last_eval=(
            last_eval
        ),
    )

    for epoch in range(
    sft_config.max_epochs
):

        epoch_loss_sum = 0.0

        epoch_batches = 0


        print()
        print("=" * 70)

        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{sft_config.max_epochs}"
        )

        print("=" * 70)


        for batch in train_loader:

            input_ids = (
                batch.input_ids
                .to(device)
            )


            targets = (
                batch.targets
                .to(device)
            )


            attention_mask = (
                batch.attention_mask
                .to(device)
            )


            metrics = (
                sft_train_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=(
                        input_ids
                    ),
                    targets=(
                        targets
                    ),
                    attention_mask=(
                        attention_mask
                    ),
                    grad_clip_norm=(
                        sft_config
                        .grad_clip_norm
                    ),
                )
            )


            completed_steps += 1

            epoch_batches += 1


            epoch_loss_sum += (
                metrics["loss"]
            )


            active_targets = (
                targets
                != -100
            ).sum().item()


            real_positions = (
                attention_mask
                .sum()
                .item()
            )


            total_positions = (
                attention_mask
                .numel()
            )


            padding_ratio = (
                1.0
                -
                (
                    real_positions
                    / total_positions
                )
            )


            print(
                f"Step "
                f"{completed_steps:4d} | "
                f"batch "
                f"{input_ids.shape[0]}x"
                f"{input_ids.shape[1]} | "
                f"targets "
                f"{active_targets:3d} | "
                f"pad "
                f"{padding_ratio:.1%} | "
                f"loss "
                f"{metrics['loss']:.4f} | "
                f"grad "
                f"{metrics['gradient_norm']:.4f}"
            )


        completed_epochs += 1


        mean_train_loss = (
            epoch_loss_sum
            / epoch_batches
        )


        val_metrics = (
            evaluate_sft_loss(
                model=model,
                dataset=val_dataset,
                tokenizer=tokenizer,
                batch_size=(
                    sft_config.batch_size
                ),
                device=device,
            )
        )


        last_eval = {
            "step": (
                completed_steps
            ),
            "epoch": (
                completed_epochs
            ),
            "train_loss": (
                mean_train_loss
            ),
            "val": (
                val_metrics
            ),
        }


        print()
        print(
            f"EPOCH "
            f"{completed_epochs} "
            f"SUMMARY | "
            f"train loss "
            f"{mean_train_loss:.4f} | "
            f"val loss "
            f"{val_metrics['loss']:.4f} | "
            f"val ppl "
            f"{val_metrics['perplexity']:.2f}"
        )


        is_new_best = (
            val_metrics["loss"]
            <
            best_val_loss
        )


        if is_new_best:

            best_val_loss = (
                val_metrics["loss"]
            )


            save_chat_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                sft_config=sft_config,
                completed_steps=(
                    completed_steps
                ),
                completed_epochs=(
                    completed_epochs
                ),
                best_val_loss=(
                    best_val_loss
                ),
                tokenizer_hash=(
                    tokenizer_hash
                ),
                base_checkpoint_path=(
                    run_config
                    .base_checkpoint
                ),
                base_checkpoint_hash=(
                    base_checkpoint_hash
                ),
                last_eval=(
                    last_eval
                ),
            )


            print(
                "New best chat checkpoint | "
                f"val loss "
                f"{best_val_loss:.4f}"
            )


        save_chat_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            sft_config=sft_config,
            completed_steps=(
                completed_steps
            ),
            completed_epochs=(
                completed_epochs
            ),
            best_val_loss=(
                best_val_loss
            ),
            tokenizer_hash=(
                tokenizer_hash
            ),
            base_checkpoint_path=(
                run_config
                .base_checkpoint
            ),
            base_checkpoint_hash=(
                base_checkpoint_hash
            ),
            last_eval=(
                last_eval
            ),
        )
    # for epoch in range(
    #     sft_config.max_epochs
    # ):

    #     indices = list(
    #         range(
    #             len(train_dataset)
    #         )
    #     )


    #     random.shuffle(
    #         indices
    #     )


    #     epoch_losses = []


    #     print()
    #     print("=" * 70)

    #     print(
    #         f"EPOCH "
    #         f"{epoch + 1}/"
    #         f"{sft_config.max_epochs}"
    #     )

    #     print("=" * 70)


    #     for batch in train_loader:
    #         input_ids = (
    #             batch
    #             .input_ids
    #             .to(device)
    #         )


    #         targets = (
    #             batch
    #             .targets
    #             .to(device)
    #         )


    #         attention_mask = (
    #             batch
    #             .attention_mask
    #             .to(device)
    #         )

    #         # example = (
    #         #     train_dataset[
    #         #         example_index
    #         #     ]
    #         # )


    #         # input_ids = (
    #         #     example
    #         #     .input_ids
    #         #     .unsqueeze(0)
    #         #     .to(device)
    #         # )


    #         # targets = (
    #         #     example
    #         #     .targets
    #         #     .unsqueeze(0)
    #         #     .to(device)
    #         # )


    #         metrics = sft_train_step(
    #             model=model,
    #             optimizer=optimizer,
    #             input_ids=input_ids,
    #             targets=targets,
    #             attention_mask=(
    #                 attention_mask
    #             ),
    #             grad_clip_norm=(
    #                 sft_config
    #                 .grad_clip_norm
    #             ),
    #         )


    #         completed_steps += 1


    #         epoch_losses.append(
    #             metrics["loss"]
    #         )


    #         print(
    #             f"Step "
    #             f"{completed_steps:4d} | "
    #             f"example "
    #             f"{example_index:2d} | "
    #             f"loss "
    #             f"{metrics['loss']:.4f} | "
    #             f"grad "
    #             f"{metrics['gradient_norm']:.4f}"
    #         )


    #     completed_epochs += 1


    #     mean_train_loss = (
    #         sum(epoch_losses)
    #         / len(epoch_losses)
    #     )


    #     val_metrics = (
    #         evaluate_sft_loss(
    #             model=model,
    # dataset=val_dataset,
    # tokenizer=tokenizer,
    # batch_size=(
    #     sft_config.batch_size
    # ),
    # device=device,
    #         )
    #     )


    #     last_eval = {
    #         "step": (
    #             completed_steps
    #         ),
    #         "epoch": (
    #             completed_epochs
    #         ),
    #         "train_loss": (
    #             mean_train_loss
    #         ),
    #         "val": (
    #             val_metrics
    #         ),
    #     }


    #     print()
    #     print(
    #         f"EPOCH "
    #         f"{completed_epochs} "
    #         f"SUMMARY | "
    #         f"train loss "
    #         f"{mean_train_loss:.4f} | "
    #         f"val loss "
    #         f"{val_metrics['loss']:.4f} | "
    #         f"val ppl "
    #         f"{val_metrics['perplexity']:.2f}"
    #     )


    #     is_new_best = (
    #         val_metrics["loss"]
    #         <
    #         best_val_loss
    #     )


    #     if is_new_best:

    #         best_val_loss = (
    #             val_metrics["loss"]
    #         )


    #         save_chat_checkpoint(
    #             path=best_path,
    #             model=model,
    #             optimizer=optimizer,
    #             sft_config=sft_config,
    #             completed_steps=(
    #                 completed_steps
    #             ),
    #             completed_epochs=(
    #                 completed_epochs
    #             ),
    #             best_val_loss=(
    #                 best_val_loss
    #             ),
    #             tokenizer_hash=(
    #                 tokenizer_hash
    #             ),
    #             base_checkpoint_path=(
    #                 run_config
    #                 .base_checkpoint
    #             ),
    #             base_checkpoint_hash=(
    #                 base_checkpoint_hash
    #             ),
    #             last_eval=(
    #                 last_eval
    #             ),
    #         )


    #         print(
    #             "New best chat checkpoint | "
    #             f"val loss "
    #             f"{best_val_loss:.4f}"
    #         )


    #     #
    #     # latest_chat.pt always contains
    #     # the most recent completed epoch.
    #     #

    #     save_chat_checkpoint(
    #         path=latest_path,
    #         model=model,
    #         optimizer=optimizer,
    #         sft_config=sft_config,
    #         completed_steps=(
    #             completed_steps
    #         ),
    #         completed_epochs=(
    #             completed_epochs
    #         ),
    #         best_val_loss=(
    #             best_val_loss
    #         ),
    #         tokenizer_hash=(
    #             tokenizer_hash
    #         ),
    #         base_checkpoint_path=(
    #             run_config
    #             .base_checkpoint
    #         ),
    #         base_checkpoint_hash=(
    #             base_checkpoint_hash
    #         ),
    #         last_eval=(
    #             last_eval
    #         ),
    #     )


    # print()
    # print("=" * 70)
    # print("SFT COMPLETE")
    # print("=" * 70)

    # print()
    # print(
    #     "Best checkpoint:",
    #     best_path,
    # )

    # print(
    #     "Latest checkpoint:",
    #     latest_path,
    # )

    # print(
    #     "Best validation loss:",
    #     round(
    #         best_val_loss,
    #         4,
    #     ),
    # )
    # train_loader = (
    #     create_sft_dataloader(
    #         dataset=train_dataset,
    #         batch_size=(
    #             sft_config
    #             .batch_size
    #         ),
    #         pad_token_id=(
    #             tokenizer.eos_token_id
    #         ),
    #         shuffle=True,
    #     )
    # )


if __name__ == "__main__":
    main()