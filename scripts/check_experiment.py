import torch

from tinygpt.config import ModelConfig
from tinygpt.config import TrainingConfig
from tinygpt.utils.device import get_device
from tinygpt.utils.random import set_seed


model_config = ModelConfig()

training_config = TrainingConfig()


set_seed(training_config.seed)


device = get_device()


print("=" * 60)
print("TINYGPT EXPERIMENT")
print("=" * 60)

print()

print("Device:")
print(device)

print()

print("Model config:")
print(model_config)

print()

print("Training config:")
print(training_config)

print()

print("Head dimension:")
print(model_config.head_dim)

print()

print("Random tensor:")
print(torch.randn(5))