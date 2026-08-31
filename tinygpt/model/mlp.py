import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0"
            )

        if d_ff <= 0:
            raise ValueError(
                "d_ff must be greater than 0"
            )

        self.d_model = d_model
        self.d_ff = d_ff


        self.gate_projection = nn.Linear(
            in_features=d_model,
            out_features=d_ff,
            bias=False,
        )


        self.up_projection = nn.Linear(
            in_features=d_model,
            out_features=d_ff,
            bias=False,
        )


        self.down_projection = nn.Linear(
            in_features=d_ff,
            out_features=d_model,
            bias=False,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.d_model:
            raise ValueError(
                "Last dimension of x "
                "must equal d_model"
            )


        gate = self.gate_projection(
            x
        )


        gate = F.silu(
            gate
        )


        up = self.up_projection(
            x
        )


        hidden = (
            gate
            * up
        )


        output = (
            self.down_projection(
                hidden
            )
        )


        return output
    
    
    
    
    # def forward(
    #         self,
    #         x: torch.Tensor,
    #     ) -> torch.Tensor:
    
    #         gate_pre_activation = (
    #             mlp.gate_projection(x)
    #         )

    #         gate = torch.nn.functional.silu(
    #             gate_pre_activation
    #         )

    #         up = mlp.up_projection(x)

    #         hidden = gate * up


    #         print()
    #         print("Gate pre-activation:")
    #         print(
    #             gate_pre_activation.shape
    #         )


    #         print()
    #         print("Gate after SiLU:")
    #         print(
    #             gate.shape
    #         )


    #         print()
    #         print("Up projection:")
    #         print(
    #             up.shape
    #         )


    #         print()
    #         print("Gated hidden:")
    #         print(
    #             hidden.shape
    #         )