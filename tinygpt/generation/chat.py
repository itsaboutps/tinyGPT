from tinygpt.sft.formatting import (
    ChatMessage,
    format_chat_history,
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


    if not messages:

        raise ValueError(
            "At least one user message "
            "is required"
        )


    #
    # Current conversation must end
    # with a user message because we're
    # about to generate the assistant.
    #

    if (
        messages[-1].role.lower()
        != "user"
    ):

        raise ValueError(
            "Conversation must end "
            "with a user message"
        )


    #
    # Always preserve the latest user
    # message.
    #

    current_user = (
        messages[-1]
    )


    selected = [
        current_user
    ]


    prompt = format_chat_history(
        system=system,
        messages=selected,
        add_assistant_prompt=True,
    )


    token_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    if len(token_ids) > prompt_budget:

        raise ValueError(
            "Current user message and "
            "system prompt exceed the "
            "available context window"
        )


    #
    # Everything before the latest
    # user message should consist of
    # complete:
    #
    # User → Assistant
    #
    # turns.
    #

    previous = (
        messages[:-1]
    )


    if len(previous) % 2 != 0:

        raise ValueError(
            "Previous conversation history "
            "must contain complete "
            "user/assistant turns"
        )


    previous_turns = []


    for index in range(
        0,
        len(previous),
        2,
    ):

        user_message = (
            previous[index]
        )

        assistant_message = (
            previous[index + 1]
        )


        if (
            user_message.role.lower()
            != "user"
            or
            assistant_message.role.lower()
            != "assistant"
        ):

            raise ValueError(
                "Conversation history must "
                "alternate user and assistant"
            )


        previous_turns.append(
            [
                user_message,
                assistant_message,
            ]
        )


    #
    # Add the newest completed turns
    # first.
    #
    # Stop when an older complete turn
    # no longer fits.
    #

    selected_previous = []


    for turn in reversed(
        previous_turns
    ):

        candidate_previous = (
            turn
            +
            selected_previous
        )


        candidate_messages = (
            candidate_previous
            +
            [current_user]
        )


        candidate_prompt = (
            format_chat_history(
                system=system,
                messages=(
                    candidate_messages
                ),
                add_assistant_prompt=True,
            )
        )


        candidate_ids = (
            tokenizer.encode(
                candidate_prompt,
                add_eos=False,
            )
        )


        if (
            len(candidate_ids)
            >
            prompt_budget
        ):

            break


        selected_previous = (
            candidate_previous
        )


    final_messages = (
        selected_previous
        +
        [current_user]
    )


    return format_chat_history(
        system=system,
        messages=final_messages,
        add_assistant_prompt=True,
    )


def clean_chat_response(
    text: str,
) -> str:

    stop_markers = [
        "### User:",
        "### System:",
        "### Assistant:",
    ]


    cleaned = text


    for marker in stop_markers:

        if marker in cleaned:

            cleaned = (
                cleaned.split(
                    marker,
                    1,
                )[0]
            )


    return cleaned.strip()