import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0"
            )

        if eps <= 0:
            raise ValueError(
                "eps must be greater than 0"
            )

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.d_model:
            raise ValueError(
                "Last dimension of x must "
                "equal d_model"
            )

        mean_square = (
            x
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        rms = torch.sqrt(
            mean_square
            + self.eps
        )

        normalized = (
            x / rms
        )

        return (
            normalized
            * self.weight
        )