from tinygpt.config import (
    ModelConfig,
    TrainingConfig,
)
from tinygpt.model.gpt import (
    TinyGPT,
)
from tinygpt.training.optimizer import (
    create_optimizer,
)


model_config = ModelConfig(
    vocab_size=1024
)


training_config = TrainingConfig()


model = TinyGPT(
    model_config
)


optimizer = create_optimizer(
    model=model,
    config=training_config,
)


print(
    "Number of parameter groups:"
)

print(
    len(
        optimizer.param_groups
    )
)


for index, group in enumerate(
    optimizer.param_groups
):

    parameter_count = sum(
        parameter.numel()
        for parameter
        in group["params"]
    )

    print()
    print(
        f"Group {index}"
    )

    print(
        "Weight decay:",
        group[
            "weight_decay"
        ],
    )

    print(
        "Parameters:",
        parameter_count,
    )
    
model_parameter_ids = {
    id(parameter)
    for parameter
    in model.parameters()
    if parameter.requires_grad
}


optimizer_parameter_ids = []


for group in optimizer.param_groups:

    for parameter in group["params"]:

        optimizer_parameter_ids.append(
            id(parameter)
        )


print()
print(
    "Unique optimizer parameters:"
)

print(
    len(
        set(
            optimizer_parameter_ids
        )
    )
)


print()
print(
    "Model trainable parameter tensors:"
)

print(
    len(
        model_parameter_ids
    )
)


print()
print(
    "All parameters covered exactly once:"
)

print(
    (
        len(
            optimizer_parameter_ids
        )
        ==
        len(
            set(
                optimizer_parameter_ids
            )
        )
        ==
        len(
            model_parameter_ids
        )
    )
)