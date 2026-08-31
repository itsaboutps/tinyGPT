import torch
import torch.nn as nn


class RotaryPositionEmbedding(
    nn.Module
):

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even"
            )

        self.head_dim = head_dim
        self.base = base

        dimension_indices = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
        )

        inverse_frequencies = (
            1.0
            /
            (
                base
                ** (
                    dimension_indices
                    / head_dim
                )
            )
        )

        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )
        
    def get_cos_sin(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        )

        angles = (
            positions[:, None]
            *
            self.inverse_frequencies[
                None, :
            ]
        )

        cos = torch.cos(
            angles
        ).to(dtype=dtype)

        sin = torch.sin(
            angles
        ).to(dtype=dtype)

        return cos, sin
    
    
    
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.head_dim:
            raise ValueError(
                "Last tensor dimension "
                "must equal head_dim"
            )

        sequence_length = (
            x.shape[-2]
        )

        cos, sin = self.get_cos_sin(
            sequence_length=(
                sequence_length
            ),
            device=x.device,
            dtype=x.dtype,
        )

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            -
            x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            +
            x_odd * cos
        )

        rotated = torch.stack(
            [
                rotated_even,
                rotated_odd,
            ],
            dim=-1,
        )

        return rotated.flatten(
            start_dim=-2
        )