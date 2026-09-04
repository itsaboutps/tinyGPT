import torch
import torch.nn as nn

from tinygpt.config import ModelConfig
from tinygpt.model.block import (
    TransformerBlock,
)


class TransformerStack(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config
                )
                for _ in range(
                    config.n_layers
                )
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        for block in self.blocks:

            x = block(
                x,
                attention_mask=(
                    attention_mask
                ),
            )


        return x