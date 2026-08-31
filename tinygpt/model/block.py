import torch
import torch.nn as nn

from tinygpt.config import ModelConfig
from tinygpt.model.attention import (
    MultiHeadCausalSelfAttention,
)
from tinygpt.model.mlp import SwiGLU
from tinygpt.model.norm import RMSNorm



class TransformerBlock(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_norm_eps,
        )

        self.attention = (
            MultiHeadCausalSelfAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                context_length=(
                    config.context_length
                ),
                rope_base=config.rope_base,
            )
        )

        self.mlp_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_norm_eps,
        )

        self.mlp = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = (
            x
            +
            self.attention(
                self.attention_norm(x)
            )
        )

        x = (
            x
            +
            self.mlp(
                self.mlp_norm(x)
            )
        )

        return x