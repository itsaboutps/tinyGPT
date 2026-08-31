import torch


def apply_top_k(
    logits: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:

    if top_k is None:
        return logits

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )


    top_k = min(
        top_k,
        logits.shape[-1],
    )


    threshold = torch.topk(
        logits,
        k=top_k,
        dim=-1,
    ).values[
        ...,
        -1,
        None,
    ]


    return logits.masked_fill(
        logits < threshold,
        float("-inf"),
    )
    
def apply_top_p(
    logits: torch.Tensor,
    top_p: float | None,
) -> torch.Tensor:

    if top_p is None:
        return logits

    if not (
        0.0
        < top_p
        <= 1.0
    ):
        raise ValueError(
            "top_p must be in (0, 1]"
        )


    sorted_logits, sorted_indices = (
        torch.sort(
            logits,
            descending=True,
            dim=-1,
        )
    )


    sorted_probabilities = (
        torch.softmax(
            sorted_logits,
            dim=-1,
        )
    )


    cumulative_probabilities = (
        torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )
    )


    tokens_to_remove = (
        cumulative_probabilities
        > top_p
    )


    tokens_to_remove[
        ...,
        1:
    ] = (
        tokens_to_remove[
            ...,
            :-1
        ].clone()
    )


    tokens_to_remove[
        ...,
        0
    ] = False


    sorted_logits = (
        sorted_logits.masked_fill(
            tokens_to_remove,
            float("-inf"),
        )
    )


    filtered_logits = (
        torch.full_like(
            logits,
            float("-inf"),
        )
    )


    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )


    return filtered_logits


    tokens_to_remove[..., 1:] = (
        tokens_to_remove[..., :-1].clone()
    )

    tokens_to_remove[..., 0] = False
    
    
def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    if temperature < 0:
        raise ValueError(
            "temperature cannot be negative"
        )


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


    logits = apply_top_k(
        logits,
        top_k,
    )


    logits = apply_top_p(
        logits,
        top_p,
    )


    probabilities = torch.softmax(
        logits,
        dim=-1,
    )


    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )


    return next_token