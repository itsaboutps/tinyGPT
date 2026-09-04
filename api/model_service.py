import threading

import torch

from tinygpt.generation.chat import (
    build_chat_prompt,
    clean_chat_response,
)
from tinygpt.generation.generate import (
    generate,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.sft.formatting import (
    ChatMessage,
)
from tinygpt.utils.device import (
    get_device,
)
from tinygpt.generation.stream import (
    generate_text_stream,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and concise assistant."
)


class ChatModelService:

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
    ):

        self.checkpoint_path = (
            checkpoint_path
        )

        self.tokenizer_path = (
            tokenizer_path
        )


        self.device = get_device()


        self.model = None

        self.tokenizer = None

        self.checkpoint = None


        self._generation_lock = (
            threading.Lock()
        )


    @property
    def is_loaded(
        self,
    ) -> bool:

        return (
            self.model is not None
            and
            self.tokenizer is not None
        )


    def load(
        self,
    ) -> None:

        if self.is_loaded:
            return


        (
            self.model,
            self.tokenizer,
            self.checkpoint,
        ) = load_model_for_generation(
            checkpoint_path=(
                self.checkpoint_path
            ),
            tokenizer_path=(
                self.tokenizer_path
            ),
            device=self.device,
        )


    def chat_with_history(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int = 40,
        temperature: float = 0.3,
        top_k: int = 20,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> str:

        if not self.is_loaded:

            raise RuntimeError(
                "Model is not loaded"
            )


        if not messages:

            raise ValueError(
                "messages cannot be empty"
            )


        prompt = build_chat_prompt(
            tokenizer=self.tokenizer,
            system=(
                DEFAULT_SYSTEM_PROMPT
            ),
            messages=messages,
            context_length=(
                self.model.config
                .context_length
            ),
            reserve_generation_tokens=(
                max_new_tokens
            ),
        )


        generator = (
            torch.Generator()
            .manual_seed(
                seed
            )
        )


        with self._generation_lock:

            result = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=(
                    max_new_tokens
                ),
                device=self.device,
                temperature=(
                    temperature
                ),
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )


        response = (
            clean_chat_response(
                result[
                    "completion"
                ]
            )
        )


        if not response:

            response = (
                "[No response generated]"
            )


        return response


    def chat(
        self,
        message: str,
    ) -> str:

        message = (
            message.strip()
        )


        if not message:

            raise ValueError(
                "message cannot be empty"
            )


        messages = [
            ChatMessage(
                role="user",
                content=message,
            )
        ]


        return self.chat_with_history(
            messages=messages,
        )
        
        
        
    def stream_chat_with_history(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int = 40,
        temperature: float = 0.3,
        top_k: int = 20,
        top_p: float = 0.9,
        seed: int = 42,
    ):

        if not self.is_loaded:

            raise RuntimeError(
                "Model is not loaded"
            )


        if not messages:

            raise ValueError(
                "messages cannot be empty"
            )


        prompt = build_chat_prompt(
            tokenizer=self.tokenizer,
            system=(
                DEFAULT_SYSTEM_PROMPT
            ),
            messages=messages,
            context_length=(
                self.model.config
                .context_length
            ),
            reserve_generation_tokens=(
                max_new_tokens
            ),
        )


        rng = (
            torch.Generator()
            .manual_seed(
                seed
            )
        )


        #
        # Keep the lock for the COMPLETE
        # generation.
        #
        # The lock remains held even while
        # chunks are yielded.
        #

        with self._generation_lock:

            for chunk in (
                generate_text_stream(
                    model=self.model,
                    tokenizer=(
                        self.tokenizer
                    ),
                    prompt=prompt,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                    device=self.device,
                    temperature=(
                        temperature
                    ),
                    top_k=top_k,
                    top_p=top_p,
                    generator=rng,
                )
            ):

                yield chunk