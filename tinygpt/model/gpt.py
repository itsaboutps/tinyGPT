import torch
import torch.nn as nn

from tinygpt.config import ModelConfig
from tinygpt.model.embeddings import TokenEmbedding
from tinygpt.model.norm import RMSNorm
from tinygpt.model.transformer import TransformerStack


class TinyGPT(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        
        self.token_embedding = TokenEmbedding(
                               vocab_size=config.vocab_size,
                               d_model=config.d_model,
        )

        self.transformer = TransformerStack(
            config
        )

        self.final_norm = RMSNorm(
            d_model=config.d_model,
            eps=config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False,
        )


        self.apply(
            self._init_weights
        )


        self.lm_head.weight = (
            self.token_embedding
            .embedding
            .weight
        )
        
        
    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:

        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
    
    
        
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have "
                "shape [B, T]"
            )


        if token_ids.dtype != torch.long:
            raise ValueError(
                "token_ids must use "
                "torch.long dtype"
            )


        B, T = token_ids.shape


        if T > self.config.context_length:
            raise ValueError(
                "Sequence length exceeds "
                "configured context length"
            )


        if token_ids.numel() > 0:

            minimum_id = (
                token_ids.min().item()
            )

            maximum_id = (
                token_ids.max().item()
            )

            if minimum_id < 0:
                raise ValueError(
                    "Token IDs cannot "
                    "be negative"
                )

            if (
                maximum_id
                >= self.config.vocab_size
            ):
                raise ValueError(
                    "Token ID exceeds "
                    "configured vocabulary"
                )


        x = self.token_embedding(
            token_ids
        )


        x = self.transformer(
            x
        )


        x = self.final_norm(
            x
        )


        logits = self.lm_head(
            x
        )


        return logits