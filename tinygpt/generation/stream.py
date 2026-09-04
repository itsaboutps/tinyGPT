import torch
import torch.nn.functional as F


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:

    if logits.ndim != 2:

        raise ValueError(
            "logits must have shape [B, V]"
        )


    if temperature < 0:

        raise ValueError(
            "temperature cannot be negative"
        )


    if top_k < 0:

        raise ValueError(
            "top_k cannot be negative"
        )


    if not (
        0.0
        <
        top_p
        <=
        1.0
    ):

        raise ValueError(
            "top_p must be in (0, 1]"
        )


    #
    # temperature = 0
    #
    # means deterministic greedy
    # generation.
    #

    if temperature == 0:

        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )


    logits = (
        logits
        / temperature
    )


    #
    # Top-K filtering
    #

    if top_k > 0:

        actual_k = min(
            top_k,
            logits.shape[-1],
        )


        top_values, _ = (
            torch.topk(
                logits,
                k=actual_k,
                dim=-1,
            )
        )


        cutoff = (
            top_values[
                :,
                -1,
            ]
            .unsqueeze(-1)
        )


        logits = (
            logits.masked_fill(
                logits < cutoff,
                float("-inf"),
            )
        )


    probabilities = (
        F.softmax(
            logits,
            dim=-1,
        )
    )


    #
    # Top-P / nucleus filtering
    #

    if top_p < 1.0:

        (
            sorted_probabilities,
            sorted_indices,
        ) = torch.sort(
            probabilities,
            descending=True,
            dim=-1,
        )


        cumulative_probabilities = (
            torch.cumsum(
                sorted_probabilities,
                dim=-1,
            )
        )


        remove_mask = (
            cumulative_probabilities
            >
            top_p
        )


        #
        # Keep the first token that
        # crosses the threshold.
        #

        remove_mask[
            :,
            1:
        ] = (
            remove_mask[
                :,
                :-1,
            ]
            .clone()
        )


        remove_mask[
            :,
            0,
        ] = False


        sorted_probabilities = (
            sorted_probabilities
            .masked_fill(
                remove_mask,
                0.0,
            )
        )


        sorted_probabilities = (
            sorted_probabilities
            /
            sorted_probabilities
            .sum(
                dim=-1,
                keepdim=True,
            )
        )


        sampled_sorted_index = (
            torch.multinomial(
                sorted_probabilities,
                num_samples=1,
                generator=generator,
            )
        )


        next_token = (
            sorted_indices.gather(
                dim=-1,
                index=(
                    sampled_sorted_index
                ),
            )
        )


        return next_token


    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )


def generate_text_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.3,
    top_k: int = 20,
    top_p: float = 0.9,
    generator: torch.Generator | None = None,
):

    if max_new_tokens <= 0:

        raise ValueError(
            "max_new_tokens must "
            "be greater than 0"
        )


    prompt_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    if not prompt_ids:

        raise ValueError(
            "Prompt produced no tokens"
        )


    generated = torch.tensor(
        [
            prompt_ids
        ],
        dtype=torch.long,
        device=device,
    )


    if generator is None:

        generator = (
            torch.Generator()
            .manual_seed(42)
        )


    new_token_ids = []

    emitted_text = ""


    model.eval()


    with torch.inference_mode():

        for _ in range(
            max_new_tokens
        ):

            #
            # The complete conversation may
            # eventually exceed our model's
            # context window.
            #
            # The model can only see the most
            # recent context_length tokens.
            #

            model_input = (
                generated[
                    :,
                    -model.config
                    .context_length:
                ]
            )


            logits = model(
                model_input
            )


            #
            # We only need prediction from
            # the final position.
            #

            next_token_logits = (
                logits[
                    :,
                    -1,
                    :
                ]
            )


            next_token = (
                _sample_next_token(
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


            next_token_id = int(
                next_token.item()
            )


            #
            # EOS means generation is done.
            #

            if (
                next_token_id
                ==
                tokenizer.eos_token_id
            ):

                break


            generated = torch.cat(
                [
                    generated,
                    next_token.to(
                        device=device,
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )


            new_token_ids.append(
                next_token_id
            )


            #
            # Decode ALL generated completion
            # tokens together.
            #
            # This is important for our
            # byte-level BPE tokenizer.
            #

            decoded_text = (
                tokenizer.decode(
                    new_token_ids
                )
            )


            #
            # If decoding currently ends with
            # the UTF-8 replacement character,
            # we may be in the middle of a
            # multi-byte character.
            #
            # Wait for another token.
            #

            if decoded_text.endswith(
                "\ufffd"
            ):

                continue


            #
            # Normally decoding is monotonic:
            #
            # previous:
            # "The answer"
            #
            # current:
            # "The answer is"
            #
            # We only emit:
            #
            # " is"
            #

            if not decoded_text.startswith(
                emitted_text
            ):

                continue


            delta = (
                decoded_text[
                    len(
                        emitted_text
                    ):
                ]
            )


            if delta:

                emitted_text = (
                    decoded_text
                )


                yield delta


    #
    # Final decode in case something
    # remained buffered.
    #

    final_text = tokenizer.decode(
        new_token_ids
    )


    if final_text.startswith(
        emitted_text
    ):

        remaining = (
            final_text[
                len(
                    emitted_text
                ):
            ]
        )


        if remaining:

            yield remaining