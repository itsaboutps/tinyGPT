from collections import Counter

from tinygpt.tokenizer.bpe import (
    BPETokenizer,
    MergeRule,
    merge_pair,
)
def count_pairs(
    token_ids: list[int],
) -> Counter:

    pairs = zip(
        token_ids,
        token_ids[1:],
    )

    return Counter(pairs)
class BPETrainer:

    BASE_VOCAB_SIZE = 256
    NUM_SPECIAL_TOKENS = 1

    def __init__(
        self,
        target_vocab_size: int,
        min_pair_frequency: int = 2,
    ):
        self.target_vocab_size = (
            target_vocab_size
        )

        self.min_pair_frequency = (
            min_pair_frequency
        )

    def train(
        self,
        text: str,
    ) -> BPETokenizer:

        token_ids = list(
            text.encode("utf-8")
        )

        merges = []

        next_token_id = (
            self.BASE_VOCAB_SIZE
        )

        merge_budget = (
            self.target_vocab_size
            - self.BASE_VOCAB_SIZE
            - self.NUM_SPECIAL_TOKENS
        )

        print(
            "Initial byte tokens:",
            len(token_ids),
        )

        for merge_number in range(
            merge_budget
        ):

            pair_counts = count_pairs(
                token_ids
            )

            if not pair_counts:
                break

            best_pair, frequency = max(
                pair_counts.items(),
                key=lambda item: (
                    item[1],
                    -item[0][0],
                    -item[0][1],
                ),
            )

            if (
                frequency
                < self.min_pair_frequency
            ):
                break

            new_id = next_token_id

            token_ids = merge_pair(
                token_ids=token_ids,
                pair=best_pair,
                new_id=new_id,
            )

            rule = MergeRule(
                left=best_pair[0],
                right=best_pair[1],
                new_id=new_id,
            )

            merges.append(rule)

            next_token_id += 1

            if (
                merge_number < 10
                or (merge_number + 1) % 100 == 0
            ):
                print(
                    "Merge",
                    merge_number + 1,
                    "pair",
                    best_pair,
                    "frequency",
                    frequency,
                    "new token",
                    new_id,
                )

        eos_token_id = next_token_id

        tokenizer = BPETokenizer(
            merges=merges,
            eos_token_id=eos_token_id,
        )

        print()
        print(
            "Learned merges:",
            len(merges),
        )

        print(
            "Final vocabulary size:",
            tokenizer.vocab_size,
        )

        return tokenizer