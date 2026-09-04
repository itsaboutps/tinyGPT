import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()


        if d_model <= 0:
            raise ValueError(
                "d_model must be "
                "greater than 0"
            )


        if d_ff <= 0:
            raise ValueError(
                "d_ff must be "
                "greater than 0"
            )


        self.d_model = (
            d_model
        )


        self.d_ff = (
            d_ff
        )


        #
        # Gate branch
        #
        # [B, T, C]
        # →
        # [B, T, F]
        #

        self.gate_projection = (
            nn.Linear(
                in_features=d_model,
                out_features=d_ff,
                bias=False,
            )
        )


        #
        # Up/content branch
        #
        # [B, T, C]
        # →
        # [B, T, F]
        #

        self.up_projection = (
            nn.Linear(
                in_features=d_model,
                out_features=d_ff,
                bias=False,
            )
        )


        #
        # Down projection
        #
        # [B, T, F]
        # →
        # [B, T, C]
        #

        self.down_projection = (
            nn.Linear(
                in_features=d_ff,
                out_features=d_model,
                bias=False,
            )
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if (
            x.shape[-1]
            != self.d_model
        ):
            raise ValueError(
                "Input feature dimension "
                "does not match d_model"
            )


        #
        # Gating branch
        #
        # [B, T, C]
        # →
        # [B, T, F]
        #

        gate = F.silu(
            self.gate_projection(
                x
            )
        )


        #
        # Content branch
        #
        # [B, T, C]
        # →
        # [B, T, F]
        #

        up = (
            self.up_projection(
                x
            )
        )


        #
        # SwiGLU gating
        #
        # Both:
        # [B, T, F]
        #
        # Element-wise multiplication.
        #

        hidden = (
            gate
            *
            up
        )


        #
        # Project back to
        # model dimension.
        #
        # [B, T, F]
        # →
        # [B, T, C]
        #

        output = (
            self.down_projection(
                hidden
            )
        )


        return output