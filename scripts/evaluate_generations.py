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

    device = get_device()


    model, tokenizer, checkpoint = (
        load_model_for_generation(
            checkpoint_path=(
                "checkpoints/"
                "tinystories_5mb_v1/"
                "best.pt"
            ),
            tokenizer_path=(
                "data/tokenizer/tokenizer.json"
            ),
            device=device,
        )
    )


    prompts = [
        "Once upon a time",
        "There was a little girl named",
        "One day, a small rabbit",
        "The blue robot walked into the garden",
        "A young boy found a strange box",
        "The database transaction failed because",
    ]


    for index, prompt in enumerate(
        prompts
    ):

        generator = (
            torch.Generator()
            .manual_seed(
                42 + index
            )
        )


        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=80,
            device=device,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            generator=generator,
        )


        print("=" * 70)

        print(
            f"PROMPT {index + 1}"
        )

        print("=" * 70)

        print()
        print(
            result["text"]
        )

        print()


if __name__ == "__main__":
    main()