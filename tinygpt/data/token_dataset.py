from pathlib import Path

import torch


class TokenDataset:

    def __init__(
        self,
        token_path: str,
    ):
        self.token_path = Path(
            token_path
        )

        if not self.token_path.exists():
            raise FileNotFoundError(
                f"Token dataset not found: "
                f"{self.token_path}"
            )

        self.tokens = torch.load(
            self.token_path,
            map_location="cpu",
            weights_only=True,
        )

        if self.tokens.ndim != 1:
            raise ValueError(
                "Token dataset must be "
                "a 1-D tensor"
            )

        if self.tokens.dtype != torch.long:
            raise ValueError(
                "Token dataset must use "
                "torch.long token IDs"
            )

    def __len__(self):
        return len(self.tokens)
    
    def get_window(
        self,
        start: int,
        context_length: int,
    ):
        end = (
            start
            + context_length
        )

        if start < 0:
            raise ValueError(
                "start must be >= 0"
            )

        if end >= len(self.tokens):
            raise ValueError(
                "Window exceeds dataset"
            )

        x = self.tokens[
            start:end
        ]

        y = self.tokens[
            start + 1:end + 1
        ]

        return x, y
    
    
    def get_batch(
    self,
    batch_size: int,
    context_length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
):
        max_start = (
            len(self.tokens)
            - context_length
            - 1
        )

        if max_start < 0:
            raise ValueError(
                "Dataset is too small for "
                "the requested context length"
            )

        start_positions = torch.randint(
            low=0,
            high=max_start + 1,
            size=(batch_size,),
            generator=generator,
        )

        inputs = []
        targets = []

        for start in start_positions.tolist():

            x, y = self.get_window(
                start=start,
                context_length=context_length,
            )

            inputs.append(x)
            targets.append(y)

        x_batch = torch.stack(
            inputs
        )

        y_batch = torch.stack(
            targets
        )

        return (
            x_batch.to(device),
            y_batch.to(device),
        )