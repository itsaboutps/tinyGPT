import torch
import torch.nn as nn


class LearnedPositionEmbedding(nn.Module):

    def __init__(
        self,
        context_length: int,
        d_model: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=d_model,
        )

    def forward(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        positions = torch.arange(
            sequence_length,
            device=device,
        )

        return self.embedding(
            positions
        )

class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        return self.embedding(
            token_ids
        )
        
    