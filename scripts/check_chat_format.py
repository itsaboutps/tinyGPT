from tinygpt.sft.formatting import (
    format_conversation,
    format_prompt,
)


system = (
    "You are a helpful assistant."
)

user = (
    "What is the sun?"
)

assistant = (
    "The sun is a star."
)


prompt = format_prompt(
    system=system,
    user=user,
)


full = format_conversation(
    system=system,
    user=user,
    assistant=assistant,
)


print("=" * 60)
print("PROMPT")
print("=" * 60)

print(
    repr(prompt)
)


print()
print("=" * 60)
print("FULL CONVERSATION")
print("=" * 60)

print(
    full
)