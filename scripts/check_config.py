from tinygpt.config import ModelConfig
from tinygpt.config import TrainingConfig


model_config = ModelConfig()

training_config = TrainingConfig()


print("Model configuration:")
print(model_config)

print()

print("Head dimension:")
print(model_config.head_dim)

print()

print("Training configuration:")
print(training_config)