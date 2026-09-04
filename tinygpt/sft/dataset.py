from dataclasses import dataclass
import json
from pathlib import Path

import torch

from tinygpt.tokenizer.bpe import (
    BPETokenizer,
)
from tinygpt.sft.formatting import (
    format_prompt,
)


IGNORE_INDEX = -100

@dataclass(frozen=True)
class InstructionRecord:
    system: str
    user: str
    assistant: str
    
    
@dataclass
class SFTExample:
    input_ids: torch.Tensor
    targets: torch.Tensor
    
    
    
def build_sft_example(
    tokenizer: BPETokenizer,
    system: str,
    user: str,
    assistant: str,
    context_length: int,
) -> SFTExample:

    prompt = format_prompt(
        system=system,
        user=user,
    )


    prompt_ids = tokenizer.encode(
        prompt,
        add_eos=False,
    )


    assistant_ids = tokenizer.encode(
        assistant.strip(),
        add_eos=True,
    )


    full_ids = (
        prompt_ids
        + assistant_ids
    )


    if len(full_ids) < 2:

        raise ValueError(
            "Conversation is too short"
        )


    if len(full_ids) > (
        context_length + 1
    ):

        raise ValueError(
            "Conversation exceeds "
            "the model context length"
        )


    input_ids = torch.tensor(
        full_ids[:-1],
        dtype=torch.long,
    )


    targets = torch.tensor(
        full_ids[1:],
        dtype=torch.long,
    )


    assistant_target_start = (
        len(prompt_ids)
        - 1
    )


    targets[
        :assistant_target_start
    ] = IGNORE_INDEX


    return SFTExample(
        input_ids=input_ids,
        targets=targets,
    )
    
    
    
def load_instruction_records(
    path: str | Path,
) -> list[InstructionRecord]:

    file_path = Path(
        path
    )


    if not file_path.exists():
        raise FileNotFoundError(
            f"Instruction file not found: "
            f"{file_path}"
        )


    records = []


    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue


            try:

                raw = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{file_path}:"
                    f"{line_number}"
                ) from exc


            required_fields = {
                "system",
                "user",
                "assistant",
            }


            missing = (
                required_fields
                - raw.keys()
            )


            if missing:

                raise ValueError(
                    f"Missing fields "
                    f"{sorted(missing)} at "
                    f"{file_path}:"
                    f"{line_number}"
                )


            values = {
                key: raw[key]
                for key
                in required_fields
            }


            for key, value in (
                values.items()
            ):

                if not isinstance(
                    value,
                    str,
                ):

                    raise ValueError(
                        f"{key} must be a string "
                        f"at {file_path}:"
                        f"{line_number}"
                    )

                if not value.strip():

                    raise ValueError(
                        f"{key} cannot be empty "
                        f"at {file_path}:"
                        f"{line_number}"
                    )


            records.append(
                InstructionRecord(
                    system=(
                        raw["system"]
                    ),
                    user=(
                        raw["user"]
                    ),
                    assistant=(
                        raw["assistant"]
                    ),
                )
            )


    if not records:

        raise ValueError(
            f"No instruction examples "
            f"found in {file_path}"
        )


    return records


class SFTDataset:

    def __init__(
        self,
        path: str | Path,
        tokenizer: BPETokenizer,
        context_length: int,
    ):

        self.records = (
            load_instruction_records(
                path
            )
        )

        self.tokenizer = tokenizer

        self.context_length = (
            context_length
        )


    def __len__(self) -> int:

        return len(
            self.records
        )


    def __getitem__(
        self,
        index: int,
    ) -> SFTExample:

        record = (
            self.records[index]
        )


        return build_sft_example(
            tokenizer=(
                self.tokenizer
            ),
            system=(
                record.system
            ),
            user=(
                record.user
            ),
            assistant=(
                record.assistant
            ),
            context_length=(
                self.context_length
            ),
        )