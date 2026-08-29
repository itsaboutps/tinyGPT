from tinygpt.config import ModelConfig
from tinygpt.model.gpt import TinyGPT


config = ModelConfig(
    vocab_size=1024
)


model = TinyGPT(
    config
)


embedding_weight = (
    model
    .token_embedding
    .embedding
    .weight
)


lm_head_weight = (
    model
    .lm_head
    .weight
)


print(
    "Same Python object:"
)

print(
    embedding_weight
    is lm_head_weight
)


print()
print(
    "Same memory pointer:"
)

print(
    embedding_weight.data_ptr()
    ==
    lm_head_weight.data_ptr()
)