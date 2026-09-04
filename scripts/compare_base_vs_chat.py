import torch

from tinygpt.generation.generate import (
    generate,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.sft.formatting import (
    format_prompt,
)
from tinygpt.utils.device import (
    get_device,
)


TOKENIZER_PATH = (
    "data/tokenizer/tokenizer.json"
)


BASE_CHECKPOINT = (
    "checkpoints/"
    "tinystories_5mb_v1/"
    "best.pt"
)


CHAT_CHECKPOINT = (
    "checkpoints/"
    "tinychat_sft_v1/"
    "best_chat.pt"
)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device,
    seed: int,
):

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )


    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=40,
        device=device,
        temperature=0.3,
        top_k=20,
        top_p=0.9,
        generator=generator,
    )


def main():

    device = get_device()


    base_model, base_tokenizer, _ = (
        load_model_for_generation(
            checkpoint_path=(
                BASE_CHECKPOINT
            ),
            tokenizer_path=(
                TOKENIZER_PATH
            ),
            device=device,
        )
    )


    chat_model, chat_tokenizer, _ = (
        load_model_for_generation(
            checkpoint_path=(
                CHAT_CHECKPOINT
            ),
            tokenizer_path=(
                TOKENIZER_PATH
            ),
            device=device,
        )
    )


    system = (
        "You are a helpful and concise "
        "assistant."
    )


    user = (
        "What is 2 + 2?"
    )


    prompt = format_prompt(
        system=system,
        user=user,
    )


    base_result = generate_response(
        model=base_model,
        tokenizer=base_tokenizer,
        prompt=prompt,
        device=device,
        seed=42,
    )


    chat_result = generate_response(
        model=chat_model,
        tokenizer=chat_tokenizer,
        prompt=prompt,
        device=device,
        seed=42,
    )


    print("=" * 70)
    print("PROMPT")
    print("=" * 70)

    print(prompt)


    print()
    print("=" * 70)
    print("BASE TINYGPT")
    print("=" * 70)

    print(
        base_result[
            "completion"
        ]
    )


    print()
    print("=" * 70)
    print("SFT TINYCHATGPT")
    print("=" * 70)

    print(
        chat_result[
            "completion"
        ]
    )


if __name__ == "__main__":
    main()