import torch

from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


@torch.inference_mode()
def inspect_next_tokens(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    top_n: int = 10,
) -> list[dict]:

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than 0"
        )


    token_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )


    context = x[
        :,
        -model.config.context_length:
    ]


    logits = model(
        context
    )


    next_logits = logits[
        0,
        -1,
    ]


    probabilities = torch.softmax(
        next_logits,
        dim=-1,
    )


    top_probabilities, top_ids = (
        torch.topk(
            probabilities,
            k=min(
                top_n,
                probabilities.numel(),
            ),
        )
    )


    results = []


    for probability, token_id in zip(
        top_probabilities.tolist(),
        top_ids.tolist(),
    ):

        token_text = (
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )
        )


        results.append(
            {
                "token_id": token_id,
                "token_text": token_text,
                "probability": probability,
            }
        )


    return results