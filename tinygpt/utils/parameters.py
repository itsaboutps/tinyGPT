import torch.nn as nn


def count_parameters(
    module: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
    )


def count_trainable_parameters(
    module: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )
    
    
def parameter_size_bytes(
    module: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        * parameter.element_size()
        for parameter
        in module.parameters()
    )
    
    
def parameter_size_mb(
    module: nn.Module,
) -> float:

    return (
        parameter_size_bytes(
            module
        )
        /
        (1024 ** 2)
    )