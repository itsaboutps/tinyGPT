import threading
import uuid

from tinygpt.sft.formatting import (
    ChatMessage,
)


class ConversationNotFoundError(
    KeyError
):
    pass


class ConversationService:

    def __init__(
        self,
    ):

        self._conversations: dict[
            str,
            list[ChatMessage],
        ] = {}

        self._lock = (
            threading.RLock()
        )


    def create_conversation(
        self,
    ) -> str:

        conversation_id = str(
            uuid.uuid4()
        )


        with self._lock:

            self._conversations[
                conversation_id
            ] = []


        return conversation_id


    def exists(
        self,
        conversation_id: str,
    ) -> bool:

        with self._lock:

            return (
                conversation_id
                in self._conversations
            )


    def get_messages(
        self,
        conversation_id: str,
    ) -> list[ChatMessage]:

        with self._lock:

            if (
                conversation_id
                not in self._conversations
            ):

                raise (
                    ConversationNotFoundError(
                        conversation_id
                    )
                )


            #
            # Return a copy.
            #
            # The caller must not directly
            # modify our internal list.
            #

            return list(
                self._conversations[
                    conversation_id
                ]
            )


    def append_exchange(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:

        user_message = (
            user_message.strip()
        )

        assistant_message = (
            assistant_message.strip()
        )


        if not user_message:

            raise ValueError(
                "user_message cannot be empty"
            )


        if not assistant_message:

            raise ValueError(
                "assistant_message cannot "
                "be empty"
            )


        with self._lock:

            if (
                conversation_id
                not in self._conversations
            ):

                raise (
                    ConversationNotFoundError(
                        conversation_id
                    )
                )


            messages = (
                self._conversations[
                    conversation_id
                ]
            )


            messages.append(
                ChatMessage(
                    role="user",
                    content=user_message,
                )
            )


            messages.append(
                ChatMessage(
                    role="assistant",
                    content=(
                        assistant_message
                    ),
                )
            )