import torch

from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


import torch

from tinygpt.generation.sampling import (
    sample_next_token,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    generator: torch.Generator | None = None,
) -> dict:

    if not prompt:
        raise ValueError(
            "prompt cannot be empty"
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must "
            "be greater than 0"
        )


    prompt_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )


    model.eval()


    stopped_on_eos = False


    for _ in range(
        max_new_tokens
    ):

        context = generated[
            :,
            -model.config.context_length:
        ]


        logits = model(
            context
        )


        next_token_logits = (
            logits[
                :,
                -1,
                :
            ]
        )


        next_token = (
            sample_next_token(
                logits=(
                    next_token_logits
                ),
                temperature=(
                    temperature
                ),
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
        )


        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=1,
        )


        if (
            next_token.item()
            ==
            tokenizer.eos_token_id
        ):

            stopped_on_eos = True

            break


    all_token_ids = (
        generated[
            0
        ]
        .cpu()
        .tolist()
    )


    new_token_ids = (
        all_token_ids[
            len(prompt_ids):
        ]
    )


    full_text = tokenizer.decode(
        all_token_ids,
        skip_special_tokens=True,
    )


    completion_text = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    )


    return {
        "text": full_text,
        "completion": completion_text,
        "token_ids": all_token_ids,
        "new_token_ids": new_token_ids,
        "new_tokens_generated": (
            len(new_token_ids)
        ),
        "stopped_on_eos": (
            stopped_on_eos
        ),
    }


@torch.inference_mode()
def generate_greedy(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:

    if not prompt:
        raise ValueError(
            "prompt cannot be empty"
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than 0"
        )


    prompt_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )


    model.eval()


    for _ in range(
        max_new_tokens
    ):

        context = generated[
            :,
            -model.config.context_length:
        ]


        logits = model(
            context
        )


        next_token_logits = (
            logits[
                :,
                -1,
                :
            ]
        )


        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )


        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=1,
        )


        if (
            next_token.item()
            ==
            tokenizer.eos_token_id
        ):
            break


    token_ids = (
        generated[0]
        .cpu()
        .tolist()
    )


    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )