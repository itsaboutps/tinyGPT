from tinygpt.tokenizer.bpe_trainer import (
    BPETrainer,
)


text = (
    "banana bandana banana "
    "banana bandana banana "
)

trainer = BPETrainer(
    target_vocab_size=280,
    min_pair_frequency=2,
)


tokenizer = trainer.train(
    text
)


print()
print("=" * 60)
print("BPE TEST")
print("=" * 60)


encoded = tokenizer.encode(
    text
)


decoded = tokenizer.decode(
    encoded
)


print()
print("Original:")
print(repr(text))


print()
print("Encoded:")
print(encoded)


print()
print("Original UTF-8 bytes:")
print(
    len(text.encode("utf-8"))
)


print()
print("BPE tokens:")
print(
    len(encoded)
)


print()
print("Decoded:")
print(repr(decoded))


print()
print("Round trip:")
print(
    decoded == text
)