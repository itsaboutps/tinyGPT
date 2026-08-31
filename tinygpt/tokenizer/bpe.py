from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MergeRule:
    left: int
    right: int
    new_id: int
    
def merge_pair(
    token_ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:

    result = []

    i = 0

    while i < len(token_ids):

        if (
            i < len(token_ids) - 1
            and token_ids[i] == pair[0]
            and token_ids[i + 1] == pair[1]
        ):
            result.append(new_id)

            i += 2

        else:
            result.append(
                token_ids[i]
            )

            i += 1

    return result

class BPETokenizer:

    BASE_VOCAB_SIZE = 256

    EOS_TOKEN = "<|endoftext|>"

    def __init__(
        self,
        merges: list[MergeRule],
        eos_token_id: int,
    ):
        self.merges = merges

        self.eos_token_id = (
            eos_token_id
        )

        self.merge_to_id = {
            (rule.left, rule.right):
                rule.new_id
            for rule in merges
        }

        self.merge_rank = {
            (rule.left, rule.right):
                rank
            for rank, rule
            in enumerate(merges)
        }

        self.token_bytes = {
            token_id: bytes(
                [token_id]
            )
            for token_id
            in range(
                self.BASE_VOCAB_SIZE
            )
        }

        for rule in merges:
            self.token_bytes[
                rule.new_id
            ] = (
                self.token_bytes[
                    rule.left
                ]
                +
                self.token_bytes[
                    rule.right
                ]
            )

        self.vocab_size = (
            self.eos_token_id + 1
        )
        
    def encode(
        self,
        text: str,
        add_eos: bool = False,
    ) -> list[int]:

        token_ids = list(
            text.encode("utf-8")
        )

        while len(token_ids) >= 2:

            best_pair = None
            best_rank = None

            for pair in zip(
                token_ids,
                token_ids[1:],
            ):

                rank = self.merge_rank.get(
                    pair
                )

                if rank is None:
                    continue

                if (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            new_id = self.merge_to_id[
                best_pair
            ]

            token_ids = merge_pair(
                token_ids=token_ids,
                pair=best_pair,
                new_id=new_id,
            )

        if add_eos:
            token_ids.append(
                self.eos_token_id
            )

        return token_ids
    
    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:

        parts = []

        byte_buffer = bytearray()

        for token_id in token_ids:

            if token_id == self.eos_token_id:

                if byte_buffer:
                    parts.append(
                        bytes(
                            byte_buffer
                        ).decode(
                            "utf-8",
                            errors="replace",
                        )
                    )

                    byte_buffer.clear()

                if not skip_special_tokens:
                    parts.append(
                        self.EOS_TOKEN
                    )

                continue

            token_bytes = (
                self.token_bytes.get(
                    token_id
                )
            )

            if token_bytes is None:
                raise ValueError(
                    f"Unknown token ID: "
                    f"{token_id}"
                )

            byte_buffer.extend(
                token_bytes
            )

        if byte_buffer:
            parts.append(
                bytes(
                    byte_buffer
                ).decode(
                    "utf-8",
                    errors="replace",
                )
            )

        return "".join(parts)
    
    """
    
    Wrong idea:
        token
          ↓
        decode UTF-8
          ↓
        next token
          ↓
        decode UTF-8
        
       
    Instead:
        token bytes
            +
        token bytes
            +
        token bytes
            ↓
        complete byte sequence
            ↓
        UTF-8 decode   
        
        That's why decode() uses:bytearray()
        
        
        """
    def save(
        self,
        path: str,
    ) -> None:

        file_path = Path(path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": 1,

            "type": "byte_bpe",

            "eos_token": (
                self.EOS_TOKEN
            ),

            "eos_token_id": (
                self.eos_token_id
            ),

            "merges": [
                {
                    "left": rule.left,
                    "right": rule.right,
                    "new_id": rule.new_id,
                }
                for rule in self.merges
            ],
        }

        file_path.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )
        
    @classmethod
    def load(
        cls,
        path: str,
    ) -> "BPETokenizer":

        file_path = Path(path)

        data = json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        if data["type"] != "byte_bpe":
            raise ValueError(
                "Unsupported tokenizer type"
            )

        merges = [
            MergeRule(
                left=item["left"],
                right=item["right"],
                new_id=item["new_id"],
            )
            for item in data["merges"]
        ]

        return cls(
            merges=merges,
            eos_token_id=(
                data["eos_token_id"]
            ),
        )