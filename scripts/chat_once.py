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


def main():

    device = get_device()


    model, tokenizer, checkpoint = (
        load_model_for_generation(
            checkpoint_path=(
                "checkpoints/"
                "tinychat_sft_v1/"
                "best_chat.pt"
            ),
            tokenizer_path=(
                "data/tokenizer/"
                "tokenizer.json"
            ),
            device=device,
        )
    )


    system = (
        "You are a helpful and concise "
        "assistant."
    )


    user = input(
        "You: "
    )


    prompt = format_prompt(
        system=system,
        user=user,
    )


    generator = (
        torch.Generator()
        .manual_seed(42)
    )


    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        device=device,
        temperature=0.3,
        top_k=20,
        top_p=0.9,
        generator=generator,
    )


    print()
    print(
        "TinyChatGPT:",
        result["completion"],
    )


if __name__ == "__main__":
    main()