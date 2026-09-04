from tinygpt.generation.inspect import (
    inspect_next_tokens,
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


    prompt = (
        "Once upon a time there was a little"
    )


    results = inspect_next_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        top_n=15,
    )


    print("=" * 70)
    print("NEXT TOKEN INSPECTION")
    print("=" * 70)


    print()
    print("Prompt:")
    print(
        repr(prompt)
    )


    print()
    print(
        "Checkpoint step:",
        checkpoint[
            "completed_step"
        ],
    )


    print()
    print("Top predictions:")


    for rank, item in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"id={item['token_id']:4d} | "
            f"p={item['probability']:.4f} | "
            f"{repr(item['token_text'])}"
        )


if __name__ == "__main__":
    main()