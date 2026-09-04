from dataclasses import dataclass
import math

# - Central configuration.
#    - It controls:
#        - model size
#        - training settings
#        - data paths
#        - tokenizer settings
#        - run/checkpoint identity

@dataclass
class RunConfig:
    experiment_name: str = (
        "tinystories_5mb_v1"
    )

    checkpoint_root: str = (
        "checkpoints"
    )

    resume_from: str | None = None

    def __post_init__(self):

        if not self.experiment_name.strip():
            raise ValueError(
                "experiment_name cannot be empty"
            )

        if not self.checkpoint_root.strip():
            raise ValueError(
                "checkpoint_root cannot be empty"
            )



@dataclass
class ModelConfig:
    vocab_size: int = 1024
    
    
    context_length: int = 128 
## How many tokens the model can look at at one time.Think of it as the model's "memory window".
    d_model: int = 128 
## Size of each token's embedding vector. Every token becomes a vector of 128 numbers."cat" becomes [0.23, -1.2, 0.55, ..., 128 values]
    n_heads: int = 4 
## Imagine four detectives reading the same sentence.Detective 1 looks for:Grammar
## Detective 2 looks for:Pronouns Detective 3 looks for:Meaning Detective 4 looks for:Long-distance relationships
## All of them read the same sentence but focus on different things.These detectives are called:n_heads = 4
    n_layers: int = 4
    d_ff: int = 384
    # head_dim:int = 4
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than 0"
            )

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be greater than 0"
            )

        if self.d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0"
            )

        if self.n_heads <= 0:
            raise ValueError(
                "n_heads must be greater than 0"
            )

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )
            
        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even "
                "when using RoPE"
            )
            
        if self.rms_norm_eps <= 0:
            raise ValueError(
                "rms_norm_eps must be greater than 0"
            )

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    batch_size: int = 8

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    weight_decay: float = 0.1

    grad_clip_norm: float = 1.0

    warmup_steps: int = 50

    max_steps: int = 1000

    log_interval: int = 10

    eval_interval: int = 100

    eval_batches: int = 10

    seed: int = 42
    checkpoint_interval: int = 50
    
    
    def __post_init__(self):

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0"
            )

        if self.min_learning_rate < 0:
            raise ValueError(
                "min_learning_rate cannot be negative"
            )

        if (
            self.min_learning_rate
            > self.learning_rate
        ):
            raise ValueError(
                "min_learning_rate cannot be "
                "greater than learning_rate"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if self.grad_clip_norm <= 0:
            raise ValueError(
                "grad_clip_norm must be greater than 0"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative"
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than 0"
            )

        if (
            self.warmup_steps
            >= self.max_steps
        ):
            raise ValueError(
                "warmup_steps must be smaller "
                "than max_steps"
            )

        if self.log_interval <= 0:
            raise ValueError(
                "log_interval must be greater than 0"
            )

        if self.eval_interval <= 0:
            raise ValueError(
                "eval_interval must be greater than 0"
            )

        if self.eval_batches <= 0:
            raise ValueError(
                "eval_batches must be greater than 0"
            )
            
        if self.checkpoint_interval <= 0:
            raise ValueError(
                "checkpoint_interval must be "
                "greater than 0"
            )
    
    # python -m scripts.check_config
    
    # [

    #     [0.1, 0.5, ..., 128 values], # I

    #     [0.8, 0.2, ..., 128 values], # love

    #     [0.4, 0.9, ..., 128 values] # AI

    # ]
    # Shape: (T,128) T = number of tokens
    

@dataclass
class DataConfig:
    raw_path: str = "data/raw/input.txt"
    processed_dir: str = "data/processed"

    train_fraction: float = 0.90
    val_fraction: float = 0.05
    test_fraction: float = 0.05

    def __post_init__(self):
        total = (
            self.train_fraction
            + self.val_fraction
            + self.test_fraction
        )

        if not math.isclose(total, 1.0):
            raise ValueError(
                "train_fraction + val_fraction + "
                "test_fraction must equal 1.0"
            )
            
            
@dataclass
class TokenizerConfig:
    vocab_size: int = 1024

    min_pair_frequency: int = 2

    output_path: str = (
        "data/tokenizer/tokenizer.json"
    )

    def __post_init__(self):
        if self.vocab_size < 257:
            raise ValueError(
                "vocab_size must be at least 257: "
                "256 byte tokens plus one special token"
            )

        if self.min_pair_frequency < 2:
            raise ValueError(
                "min_pair_frequency must be at least 2"
            )
            
@dataclass
class SFTConfig:

    batch_size: int = 4

    gradient_accumulation_steps: int = 4

    learning_rate: float = 5e-5

    weight_decay: float = 0.1

    grad_clip_norm: float = 1.0

    max_epochs: int = 20

    seed: int = 123


    def __post_init__(self):

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps "
                "must be greater than 0"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if self.grad_clip_norm <= 0:
            raise ValueError(
                "grad_clip_norm must be greater than 0"
            )

        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be greater than 0"
            )

@dataclass
class SFTRunConfig:

    experiment_name: str = (
        "tinychat_sft_batched_v1"
    )

    checkpoint_root: str = (
        "checkpoints"
    )

    base_checkpoint: str = (
        "checkpoints/"
        "tinystories_5mb_v1/"
        "best.pt"
    )

    resume_from: str | None = None


    def __post_init__(self):

        if not self.experiment_name.strip():
            raise ValueError(
                "experiment_name cannot be empty"
            )

        if not self.checkpoint_root.strip():
            raise ValueError(
                "checkpoint_root cannot be empty"
            )

        if not self.base_checkpoint.strip():
            raise ValueError(
                "base_checkpoint cannot be empty"
            )
    
@dataclass
class SFTRunConfig:

    experiment_name: str = (
        "tinychat_sft_v1"
    )

    checkpoint_root: str = (
        "checkpoints"
    )

    base_checkpoint: str = (
        "checkpoints/"
        "tinystories_5mb_v1/"
        "best.pt"
    )

    resume_from: str | None = None


    def __post_init__(self):

        if not self.experiment_name.strip():
            raise ValueError(
                "experiment_name cannot "
                "be empty"
            )

        if not self.checkpoint_root.strip():
            raise ValueError(
                "checkpoint_root cannot "
                "be empty"
            )

        if not self.base_checkpoint.strip():
            raise ValueError(
                "base_checkpoint cannot "
                "be empty"
            )