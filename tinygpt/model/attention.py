import torch
import torch.nn as nn

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


        self.rope = (
            RotaryPositionEmbedding(
                head_dim=head_dim,
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
            head_dim ** -0.5
        )


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
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
                "configured context length"
            )


        #
        # Q / K / V
        #

        q = self.query(
            x
        )


        k = self.key(
            x
        )


        v = self.value(
            x
        )


        #
        # RoPE expects:
        #
        # [B, H, T, D]
        #
        # We have one head, so temporarily
        # create H = 1.
        #

        q = q.unsqueeze(
            1
        )

        k = k.unsqueeze(
            1
        )


        q = self.rope(
            q
        )

        k = self.rope(
            k
        )


        q = q.squeeze(
            1
        )

        k = k.squeeze(
            1
        )


        #
        # Attention scores
        #
        # q:
        # [B, T, D]
        #
        # k^T:
        # [B, D, T]
        #
        # scores:
        # [B, T, T]
        #

        scores = (
            q
            @
            k.transpose(
                -2,
                -1,
            )
        )


        scores = (
            scores
            * self.scale
        )


        #
        # Causal mask
        #
        # Shape:
        # [T, T]
        #

        causal_mask = (
            self.causal_mask[
                :T,
                :T,
            ]
        )


        scores = (
            scores.masked_fill(
                ~causal_mask,
                float("-inf"),
            )
        )


        #
        # Padding mask
        #
        # attention_mask:
        # [B, T]
        #
        # Single-head scores:
        # [B, T, T]
        #
        # Therefore key mask:
        # [B, 1, T]
        #

        if attention_mask is not None:

            if (
                attention_mask.ndim != 2
            ):

                raise ValueError(
                    "attention_mask must "
                    "have shape [B, T]"
                )


            if attention_mask.shape != (
                B,
                T,
            ):

                raise ValueError(
                    "attention_mask must "
                    "have shape [B, T]"
                )


            attention_mask = (
                attention_mask.to(
                    device=x.device,
                    dtype=torch.bool,
                )
            )


            key_mask = (
                attention_mask[
                    :,
                    None,
                    :
                ]
            )


            scores = (
                scores.masked_fill(
                    ~key_mask,
                    float("-inf"),
                )
            )


        #
        # Convert scores into probabilities.
        #

        attention_weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )


        #
        # Weighted value aggregation
        #
        # [B, T, T]
        # @
        # [B, T, D]
        #
        # →
        #
        # [B, T, D]
        #

        output = (
            attention_weights
            @
            v
        )


        #
        # Zero output belonging to padded
        # query positions.
        #

        if attention_mask is not None:

            output = (
                output
                *
                attention_mask
                .unsqueeze(-1)
                .to(
                    output.dtype
                )
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
            d_model
            // n_heads
        )


        self.context_length = (
            context_length
        )


        if self.head_dim % 2 != 0:

            raise ValueError(
                "head_dim must be even "
                "for RoPE"
            )


        #
        # Fused:
        #
        # C → 3C
        #
        # Later split into Q, K, V.
        #

        self.qkv_projection = nn.Linear(
            in_features=d_model,
            out_features=(
                3 * d_model
            ),
            bias=False,
        )


        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )


        self.rope = (
            RotaryPositionEmbedding(
                head_dim=(
                    self.head_dim
                ),
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
            self.head_dim
            ** -0.5
        )


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
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


        #
        # Validate attention mask early.
        #

        if attention_mask is not None:

            if (
                attention_mask.ndim != 2
            ):

                raise ValueError(
                    "attention_mask must "
                    "have shape [B, T]"
                )


            if attention_mask.shape != (
                B,
                T,
            ):

                raise ValueError(
                    "attention_mask must "
                    "have shape [B, T]"
                )


            attention_mask = (
                attention_mask.to(
                    device=x.device,
                    dtype=torch.bool,
                )
            )


        #
        # Fused QKV projection
        #
        # [B, T, C]
        #
        # →
        #
        # [B, T, 3C]
        #

        qkv = (
            self.qkv_projection(
                x
            )
        )


        #
        # Each:
        #
        # [B, T, C]
        #

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )


        #
        # Split C into:
        #
        # H × D
        #
        # [B, T, C]
        #
        # →
        #
        # [B, T, H, D]
        #
        # →
        #
        # [B, H, T, D]
        #

        q = (
            q.view(
                B,
                T,
                self.n_heads,
                self.head_dim,
            )
            .transpose(
                1,
                2,
            )
        )


        k = (
            k.view(
                B,
                T,
                self.n_heads,
                self.head_dim,
            )
            .transpose(
                1,
                2,
            )
        )


        v = (
            v.view(
                B,
                T,
                self.n_heads,
                self.head_dim,
            )
            .transpose(
                1,
                2,
            )
        )


        #
        # Apply position information
        # to Q and K.
        #

        q = self.rope(
            q
        )


        k = self.rope(
            k
        )


        #
        # Attention scores
        #
        # q:
        # [B, H, T, D]
        #
        # k^T:
        # [B, H, D, T]
        #
        # scores:
        # [B, H, T, T]
        #

        scores = (
            q
            @
            k.transpose(
                -2,
                -1,
            )
        )


        scores = (
            scores
            * self.scale
        )


        #
        # Causal mask
        #
        # [T, T]
        #
        # Broadcasting turns this into:
        #
        # [B, H, T, T]
        #

        causal_mask = (
            self.causal_mask[
                :T,
                :T,
            ]
        )


        scores = (
            scores.masked_fill(
                ~causal_mask,
                float("-inf"),
            )
        )


        #
        # Padding key mask
        #
        # attention_mask:
        #
        # [B, T]
        #
        # →
        #
        # [B, 1, 1, T]
        #
        # This masks the KEY dimension
        # of:
        #
        # [B, H, T, T]
        #

        if attention_mask is not None:

            key_mask = (
                attention_mask[
                    :,
                    None,
                    None,
                    :
                ]
            )


            scores = (
                scores.masked_fill(
                    ~key_mask,
                    float("-inf"),
                )
            )


        #
        # Attention probability
        #

        attention_weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )


        #
        # Weighted Value aggregation
        #
        # [B,H,T,T]
        # @
        # [B,H,T,D]
        #
        # →
        #
        # [B,H,T,D]
        #

        output = (
            attention_weights
            @
            v
        )


        #
        # Merge attention heads
        #
        # [B,H,T,D]
        #
        # →
        #
        # [B,T,H,D]
        #
        # →
        #
        # [B,T,C]
        #

        output = (
            output
            .transpose(
                1,
                2,
            )
            .contiguous()
            .view(
                B,
                T,
                C,
            )
        )


        #
        # Final attention projection
        #

        output = (
            self.output_projection(
                output
            )
        )


        #
        # Zero outputs at padding
        # query positions.
        #

        if attention_mask is not None:

            output = (
                output
                *
                attention_mask
                .unsqueeze(-1)
                .to(
                    output.dtype
                )
            )


        #
        # Important:
        #
        # return_attention comes AFTER
        # padding-output masking.
        #
        # This means:
        #
        # return_attention=False
        #
        # and
        #
        # return_attention=True
        #
        # produce exactly the same
        # output tensor.
        #

        if return_attention:

            return (
                output,
                attention_weights,
            )


        return output