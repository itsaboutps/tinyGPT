from tinygpt.config import (
    TrainingConfig,
)
from tinygpt.training.schedule import (
    get_learning_rate,
)


config = TrainingConfig()


steps = [
    0,
    1,
    9,
    config.warmup_steps - 1,
    config.warmup_steps,
    100,
    500,
    config.max_steps - 1,
]


print("=" * 60)
print("LEARNING-RATE SCHEDULE")
print("=" * 60)


for step in steps:

    lr = get_learning_rate(
        step,
        config,
    )

    print(
        f"Step {step:4d} | "
        f"lr {lr:.8f}"
    )