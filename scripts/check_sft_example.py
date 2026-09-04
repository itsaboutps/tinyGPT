from tinygpt.config import (
    ModelConfig,
)
from tinygpt.sft.dataset import (
    build_sft_example,
    IGNORE_INDEX,
)
from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)


tokenizer = BPETokenizer.load(
    "data/tokenizer/tokenizer.json"
)


config = ModelConfig(
    vocab_size=tokenizer.vocab_size
)


example = build_sft_example(
    tokenizer=tokenizer,
    system=(
        "You are a helpful assistant."
    ),
    user=(
        "What is the sun?"
    ),
    assistant=(
        "The sun is a star."
    ),
    context_length=(
        config.context_length
    ),
)


print(
    "Input shape:",
    example.input_ids.shape,
)


print(
    "Target shape:",
    example.targets.shape,
)


print(
    "Ignored targets:",
    (
        example.targets
        == IGNORE_INDEX
    ).sum().item(),
)


print(
    "Assistant targets:",
    (
        example.targets
        != IGNORE_INDEX
    ).sum().item(),
)