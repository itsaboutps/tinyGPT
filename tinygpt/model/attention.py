import torch
import torch.nn as nn

# from tinygpt import config
from tinygpt.model.rope import (
    RotaryPositionEmbedding,
)



class CausalSelfAttentionHead(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        context_length: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.head_dim = head_dim
        self.context_length = (
            context_length
        )
        

        self.query = nn.Linear(
            in_features=d_model,
            out_features=head_dim,
            bias=False,
        )


        self.key = nn.Linear(
            in_features=d_model,
            out_features=head_dim,
            bias=False,
        )


        self.value = nn.Linear(
            in_features=d_model,
            out_features=head_dim,
            bias=False,
        )

        
        # self.rope = (
        #     RotaryPositionEmbedding(
        #         head_dim=config.head_dim,
        #         base=config.rope_base,
        #     )
        # )
        self.rope = (
                    RotaryPositionEmbedding(
                    head_dim=head_dim,
                    base=rope_base
                    )
                    )

        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )


        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )


        self.scale = (
            head_dim ** -0.5
        )
        
    def forward(
        self,
    x: torch.Tensor,
    return_attention: bool = False,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[B, T, C]"
            )


        B, T, C = x.shape


        if C != self.d_model:
            raise ValueError(
                "Input feature dimension "
                "does not match d_model"
            )


        if T > self.context_length:
            raise ValueError(
                "Sequence length exceeds "
                "configured context length"
            )


        q = self.query(x)

        k = self.key(x)

        v = self.value(x)


        q = q.unsqueeze(1)

        k = k.unsqueeze(1)


        q = self.rope(q)

        k = self.rope(k)


        q = q.squeeze(1)

        k = k.squeeze(1)


        scores = (
            q
            @
            k.transpose(-2, -1)
        )


        scores = (
            scores
            * self.scale
        )


        mask = self.causal_mask[
            :T,
            :T,
        ]


        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )


        attention_weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )


        output = (
            attention_weights
            @
            v
        )


        if return_attention:
            return (
                output,
                attention_weights,
            )

        return output
    
    
    
class MultiHeadCausalSelfAttention(
    nn.Module
):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible "
                "by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )

        self.context_length = (
            context_length
        )

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even "
                "for RoPE"
            )


        self.qkv_projection = nn.Linear(
            in_features=d_model,
            out_features=3 * d_model,
            bias=False,
        )


        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )


        self.rope = (
            RotaryPositionEmbedding(
                head_dim=self.head_dim,
                base=rope_base,
            )
        )


        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )


        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )


        self.scale = (
            self.head_dim ** -0.5
        )
        
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[B, T, C]"
            )


        B, T, C = x.shape


        if C != self.d_model:
            raise ValueError(
                "Input feature dimension "
                "does not match d_model"
            )


        if T > self.context_length:
            raise ValueError(
                "Sequence length exceeds "
                "context_length"
            )


        qkv = self.qkv_projection(
            x
        )


        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )


        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )


        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )


        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )


        q = self.rope(q)

        k = self.rope(k)


        scores = (
            q
            @
            k.transpose(-2, -1)
        )


        scores = (
            scores
            * self.scale
        )


        mask = self.causal_mask[
            :T,
            :T,
        ]


        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )


        attention_weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )


        output = (
            attention_weights
            @
            v
        )


        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(
                B,
                T,
                C,
            )
        )


        output = (
            self.output_projection(
                output
            )
        )


        if return_attention:
            return (
                output,
                attention_weights,
            )


        return output    