import torch

from tinygpt.generation.generate import (
    generate,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.sft.formatting import (
    ChatMessage,
    format_chat_history,
)
from tinygpt.utils.device import (
    get_device,
)


SYSTEM = (
    "You are a helpful and concise assistant."
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


    history = []


    print("=" * 70)
    print("TINYCHATGPT")
    print("=" * 70)

    print(
        "Type 'exit' to quit."
    )


    turn = 0


    while True:

        print()

        user = input(
            "You: "
        ).strip()


        if not user:
            continue


        if user.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye."
            )

            break


        history.append(
            ChatMessage(
                role="user",
                content=user,
            )
        )


        prompt = (
            format_chat_history(
                system=SYSTEM,
                messages=history,
                add_assistant_prompt=True,
            )
        )


        generator = (
            torch.Generator()
            .manual_seed(
                42 + turn
            )
        )


        result = generate(
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


        response = (
            result["completion"]
            .strip()
        )


        print(
            "TinyChatGPT:",
            response,
        )


        history.append(
            ChatMessage(
                role="assistant",
                content=response,
            )
        )


        turn += 1


if __name__ == "__main__":
    main()
    
    
    
from tinygpt.sft.formatting import (
    ChatMessage,
    format_chat_history,
)

from tinygpt.generation.chat import (
    build_chat_prompt,
)

def build_chat_prompt(
    tokenizer,
    system: str,
    messages: list[ChatMessage],
    context_length: int,
    reserve_generation_tokens: int,
) -> str:

    if reserve_generation_tokens <= 0:
        raise ValueError(
            "reserve_generation_tokens "
            "must be positive"
        )


    prompt_budget = (
        context_length
        - reserve_generation_tokens
    )


    if prompt_budget <= 0:
        raise ValueError(
            "Generation reserve exceeds "
            "context length"
        )


    selected = []


    for message in reversed(
        messages
    ):

        candidate = [
            message,
        ] + selected


        prompt = build_chat_prompt(
            tokenizer=tokenizer,
            system=SYSTEM,
            messages=history,
            context_length=(
                model.config
                .context_length
            ),
            reserve_generation_tokens=40,
        )


        token_ids = tokenizer.encode(
            prompt,
            add_eos=False,
        )


        if len(token_ids) > (
            prompt_budget
        ):

            break


        selected = candidate


    prompt = format_chat_history(
        system=system,
        messages=selected,
        add_assistant_prompt=True,
    )


    token_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    if len(token_ids) > (
        prompt_budget
    ):

        raise ValueError(
            "System prompt and latest "
            "conversation cannot fit "
            "within context budget"
        )


    return prompt