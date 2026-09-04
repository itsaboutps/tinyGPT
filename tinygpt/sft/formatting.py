from dataclasses import dataclass


def format_conversation(
    system: str,
    user: str,
    assistant: str,
) -> str:

    if not system.strip():
        raise ValueError(
            "system cannot be empty"
        )

    if not user.strip():
        raise ValueError(
            "user cannot be empty"
        )

    if not assistant.strip():
        raise ValueError(
            "assistant cannot be empty"
        )


    return (
        "### System:\n"
        f"{system.strip()}\n\n"
        "### User:\n"
        f"{user.strip()}\n\n"
        "### Assistant:\n"
        f"{assistant.strip()}"
    )
    
    
def format_prompt(
    system: str,
    user: str,
) -> str:

    if not system.strip():
        raise ValueError(
            "system cannot be empty"
        )

    if not user.strip():
        raise ValueError(
            "user cannot be empty"
        )


    return (
        "### System:\n"
        f"{system.strip()}\n\n"
        "### User:\n"
        f"{user.strip()}\n\n"
        "### Assistant:\n"
    )

@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str    
    
def format_chat_history(
    system: str,
    messages: list[ChatMessage],
    add_assistant_prompt: bool = True,
) -> str:

    if not system.strip():
        raise ValueError(
            "system cannot be empty"
        )


    parts = [
        "### System:\n",
        system.strip(),
        "\n\n",
    ]


    for message in messages:

        role = (
            message.role
            .strip()
            .lower()
        )


        content = (
            message.content
            .strip()
        )


        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                f"Unsupported role: "
                f"{message.role}"
            )


        if not content:
            raise ValueError(
                "message content "
                "cannot be empty"
            )


        if role == "user":

            parts.extend(
                [
                    "### User:\n",
                    content,
                    "\n\n",
                ]
            )

        else:

            parts.extend(
                [
                    "### Assistant:\n",
                    content,
                    "\n\n",
                ]
            )


    if add_assistant_prompt:

        parts.append(
            "### Assistant:\n"
        )


    return "".join(
        parts
    )