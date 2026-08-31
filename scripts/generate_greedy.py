from tinygpt.generation.generate import (
    generate_greedy,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.utils.device import (
    get_device,
)


def main():

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
    print("TINYGPT GENERATION")
    print("=" * 70)


    print()
    print("Device:")
    print(device)


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


    prompt = (
        "Once upon a time"
    )


    output = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        device=device,
    )


    print()
    print("Prompt:")
    print(prompt)


    print()
    print("Generated:")
    print(output)


if __name__ == "__main__":
    main()