import argparse

import torch

from tinygpt.generation.generate import (
    generate,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.utils.device import (
    get_device,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate text with TinyGPT"
        )
    )


    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )


    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )


    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "tinystories_5mb_v1/"
            "best.pt"
        ),
    )


    args = parser.parse_args()


    device = get_device()


    model, tokenizer, checkpoint = (
        load_model_for_generation(
            checkpoint_path=(
                args.checkpoint
            ),
            tokenizer_path=(
                "data/tokenizer/"
                "tokenizer.json"
            ),
            device=device,
        )
    )


    if device.type == "cpu":

        generator = (
            torch.Generator(
                device="cpu"
            )
            .manual_seed(
                args.seed
            )
        )

    else:

        generator = None

        torch.manual_seed(
            args.seed
        )


    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=(
            args.max_new_tokens
        ),
        device=device,
        temperature=(
            args.temperature
        ),
        top_k=args.top_k,
        top_p=args.top_p,
        generator=generator,
    )


    print("=" * 70)
    print("TINYGPT GENERATION")
    print("=" * 70)


    print()
    print("Device:")
    print(device)


    print()
    print("Checkpoint:")
    print(
        args.checkpoint
    )


    print()
    print("Checkpoint step:")
    print(
        checkpoint[
            "completed_step"
        ]
    )


    print()
    print("Best validation loss:")
    print(
        checkpoint[
            "best_val_loss"
        ]
    )


    print()
    print("Temperature:")
    print(
        args.temperature
    )


    print()
    print("Top-k:")
    print(
        args.top_k
    )


    print()
    print("Top-p:")
    print(
        args.top_p
    )


    print()
    print("Generated new tokens:")
    print(
        result[
            "new_tokens_generated"
        ]
    )


    print()
    print("Stopped on EOS:")
    print(
        result[
            "stopped_on_eos"
        ]
    )


    print()
    print("=" * 70)
    print("TEXT")
    print("=" * 70)


    print()
    print(
        result["text"]
    )


if __name__ == "__main__":
    main()