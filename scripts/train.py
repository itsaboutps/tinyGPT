import time
from pathlib import Path

import torch

from tinygpt.config import (
    ModelConfig,
    RunConfig,
    TrainingConfig,
)
from tinygpt.data.token_dataset import TokenDataset
from tinygpt.model.gpt import TinyGPT
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    write_run_metadata,
)
from tinygpt.training.evaluate import evaluate_loss
from tinygpt.training.optimizer import create_optimizer
from tinygpt.training.schedule import (
    get_learning_rate,
    set_learning_rate,
)
from tinygpt.training.step import train_step
from tinygpt.utils.device import get_device
from tinygpt.utils.hashing import file_sha256
from tinygpt.utils.parameters import (
    count_trainable_parameters,
    parameter_size_mb,
)
from tinygpt.utils.random import set_seed


def main():
    training_config = TrainingConfig()
    run_config = RunConfig()

    run_dir = (
        Path(run_config.checkpoint_root)
        / run_config.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    metadata_path = run_dir / "run_metadata.json"

    # Prevent accidentally overwriting an existing experiment.
    if run_config.resume_from is None and any(
        path.exists()
        for path in (
            latest_path,
            best_path,
            metadata_path,
        )
    ):
        raise FileExistsError(
            "This experiment directory already contains run artifacts. "
            "Set RunConfig.resume_from to latest.pt to continue it, "
            "or choose a new experiment_name."
        )

    set_seed(training_config.seed)
    device = get_device()

    tokenizer_path = Path("data/tokenizer/tokenizer.json")
    token_metadata_path = Path("data/tokens/metadata.json")

    tokenizer = BPETokenizer.load(str(tokenizer_path))
    tokenizer_hash = file_sha256(tokenizer_path)
    data_metadata_hash = file_sha256(token_metadata_path)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size
    )

    train_dataset = TokenDataset(
        "data/tokens/train.pt"
    )
    val_dataset = TokenDataset(
        "data/tokens/val.pt"
    )

    minimum_tokens = model_config.context_length + 1

    if len(train_dataset) < minimum_tokens:
        raise ValueError(
            "Training dataset is too small "
            "for configured context length"
        )

    if len(val_dataset) < minimum_tokens:
        raise ValueError(
            "Validation dataset is too small "
            "for configured context length"
        )

    model = TinyGPT(model_config).to(device)
    optimizer = create_optimizer(
        model=model,
        config=training_config,
    )

    training_generator = (
        torch.Generator()
        .manual_seed(training_config.seed + 100)
    )

    print("=" * 70)
    print("TINYGPT PRETRAINING")
    print("=" * 70)
    print()
    print("Device:")
    print(device)
    print()
    print("Vocabulary size:")
    print(model_config.vocab_size)
    print()
    print("Context length:")
    print(model_config.context_length)
    print()
    print("Layers:")
    print(model_config.n_layers)
    print()
    print("Trainable parameters:")
    print(count_trainable_parameters(model))
    print()
    print("Parameter memory MB:")
    print(round(parameter_size_mb(model), 3))
    print()
    print("Training tokens:")
    print(len(train_dataset))
    print()
    print("Validation tokens:")
    print(len(val_dataset))

    # These values must exist before the training loop begins.
    start_step = 0
    best_val_loss = float("inf")
    last_eval = None

    if run_config.resume_from is not None:
        checkpoint = load_training_checkpoint(
            path=run_config.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            model_config=model_config,
            training_config=training_config,
            tokenizer_sha256=tokenizer_hash,
            token_data_metadata_sha256=data_metadata_hash,
            training_generator=training_generator,
        )

        start_step = checkpoint["completed_step"]
        best_val_loss = checkpoint["best_val_loss"]
        last_eval = checkpoint["last_eval"]

        print()
        print("Resumed from:")
        print(run_config.resume_from)
        print("Completed steps:")
        print(start_step)
        print("Best validation loss:")
        print(best_val_loss)

    else:
        write_run_metadata(
            path=metadata_path,
            run_config=run_config,
            model_config=model_config,
            training_config=training_config,
            tokenizer_sha256=tokenizer_hash,
            token_data_metadata_sha256=data_metadata_hash,
        )

        print()
        print("=" * 70)
        print("INITIAL EVALUATION")
        print("=" * 70)

        initial_train = evaluate_loss(
            model=model,
            dataset=train_dataset,
            batch_size=training_config.batch_size,
            context_length=model_config.context_length,
            num_batches=training_config.eval_batches,
            device=device,
            seed=training_config.seed + 1000,
        )

        initial_val = evaluate_loss(
            model=model,
            dataset=val_dataset,
            batch_size=training_config.batch_size,
            context_length=model_config.context_length,
            num_batches=training_config.eval_batches,
            device=device,
            seed=training_config.seed + 2000,
        )

        print(
            f"Train loss {initial_train['loss']:.4f} | "
            f"train ppl {initial_train['perplexity']:.2f}"
        )
        print(
            f"Val loss {initial_val['loss']:.4f} | "
            f"val ppl {initial_val['perplexity']:.2f}"
        )

        best_val_loss = initial_val["loss"]
        last_eval = {
            "step": 0,
            "train": initial_train,
            "val": initial_val,
        }

        # A fresh run starts with a recoverable step-0 checkpoint.
        save_training_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            completed_step=0,
            best_val_loss=best_val_loss,
            model_config=model_config,
            training_config=training_config,
            tokenizer_sha256=tokenizer_hash,
            token_data_metadata_sha256=data_metadata_hash,
            training_generator=training_generator,
            last_eval=last_eval,
        )

        save_training_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            completed_step=0,
            best_val_loss=best_val_loss,
            model_config=model_config,
            training_config=training_config,
            tokenizer_sha256=tokenizer_hash,
            token_data_metadata_sha256=data_metadata_hash,
            training_generator=training_generator,
            last_eval=last_eval,
        )

    if start_step >= training_config.max_steps:
        print()
        print("Training is already complete.")
        print("Completed steps:")
        print(start_step)
        return

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    interval_start = time.perf_counter()
    interval_tokens = 0

    for step in range(
        start_step,
        training_config.max_steps,
    ):
        learning_rate = get_learning_rate(
            step,
            training_config,
        )
        set_learning_rate(
            optimizer,
            learning_rate,
        )

        x, y = train_dataset.get_batch(
            batch_size=training_config.batch_size,
            context_length=model_config.context_length,
            device=device,
            generator=training_generator,
        )

        metrics = train_step(
            model=model,
            optimizer=optimizer,
            x=x,
            y=y,
            grad_clip_norm=training_config.grad_clip_norm,
        )

        interval_tokens += x.numel()
        completed_step = step + 1

        should_log = (
            completed_step % training_config.log_interval == 0
            or completed_step == 1
            or completed_step == training_config.max_steps
        )

        if should_log:
            elapsed = time.perf_counter() - interval_start
            tokens_per_second = interval_tokens / elapsed

            print(
                f"Step {completed_step:4d}/"
                f"{training_config.max_steps} | "
                f"loss {metrics['loss']:.4f} | "
                f"lr {learning_rate:.8f} | "
                f"grad {metrics['gradient_norm']:.4f} | "
                f"tok/s {tokens_per_second:.1f}"
            )

            interval_start = time.perf_counter()
            interval_tokens = 0

        should_evaluate = (
            completed_step % training_config.eval_interval == 0
            or completed_step == training_config.max_steps
        )

        if should_evaluate:
            train_metrics = evaluate_loss(
                model=model,
                dataset=train_dataset,
                batch_size=training_config.batch_size,
                context_length=model_config.context_length,
                num_batches=training_config.eval_batches,
                device=device,
                seed=training_config.seed + 1000,
            )

            val_metrics = evaluate_loss(
                model=model,
                dataset=val_dataset,
                batch_size=training_config.batch_size,
                context_length=model_config.context_length,
                num_batches=training_config.eval_batches,
                device=device,
                seed=training_config.seed + 2000,
            )

            last_eval = {
                "step": completed_step,
                "train": train_metrics,
                "val": val_metrics,
            }

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

                save_training_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    completed_step=completed_step,
                    best_val_loss=best_val_loss,
                    model_config=model_config,
                    training_config=training_config,
                    tokenizer_sha256=tokenizer_hash,
                    token_data_metadata_sha256=data_metadata_hash,
                    training_generator=training_generator,
                    last_eval=last_eval,
                )

                print(
                    f"New best checkpoint | "
                    f"val loss {best_val_loss:.4f}"
                )

            print("-" * 70)
            print(
                f"EVAL step {completed_step} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train ppl {train_metrics['perplexity']:.2f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"val ppl {val_metrics['perplexity']:.2f}"
            )
            print("-" * 70)

        # Save latest only after evaluation/best-model bookkeeping so the
        # checkpoint contains the newest best_val_loss and last_eval values.
        should_checkpoint = (
            completed_step % training_config.checkpoint_interval == 0
            or completed_step == training_config.max_steps
        )

        if should_checkpoint:
            save_training_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                completed_step=completed_step,
                best_val_loss=best_val_loss,
                model_config=model_config,
                training_config=training_config,
                tokenizer_sha256=tokenizer_hash,
                token_data_metadata_sha256=data_metadata_hash,
                training_generator=training_generator,
                last_eval=last_eval,
            )

        # Exclude evaluation/checkpoint overhead from the next throughput
        # interval as much as possible.
        if should_evaluate or should_checkpoint:
            interval_start = time.perf_counter()
            interval_tokens = 0

    print()
    print("Training complete.")
    print("Best validation loss:")
    print(best_val_loss)
    print("Latest checkpoint:")
    print(latest_path)
    print("Best checkpoint:")
    print(best_path)


if __name__ == "__main__":
    main()
