import torch

from tinygpt.generation.generate import (
    generate,
)
from tinygpt.generation.load import (
    load_model_for_generation,
)
from tinygpt.utils.device import (
    get_device,
)


device = get_device()


model, tokenizer, _ = (
    load_model_for_generation(
        checkpoint_path=(
            "checkpoints/"
            "tinystories_5mb_v1/"
            "best.pt"
        ),
        tokenizer_path=(
            "data/tokenizer/"
            "tokenizer.json"
        ),
        device=device,
    )
)


generator_1 = (
    torch.Generator()
    .manual_seed(42)
)


generator_2 = (
    torch.Generator()
    .manual_seed(42)
)


result_1 = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=20,
    device=device,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    generator=generator_1,
)


result_2 = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=20,
    device=device,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    generator=generator_2,
)


print(
    "Same seed gives same tokens:"
)

print(
    result_1["token_ids"]
    ==
    result_2["token_ids"]
)


print()
print(
    result_1["text"]
)