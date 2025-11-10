"""
Configuration management for IceNet
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Model configuration"""
    type: str = "transformer"  # transformer, cnn, rnn, lstm, gru
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    vocab_size: int = 50000
    max_seq_length: int = 512
    activation: str = "gelu"
    use_mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # linear, cosine, constant
    accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    """Data configuration"""
    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: Optional[str] = None
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    shuffle: bool = True
    max_samples: Optional[int] = None


@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "auto"  # auto, mps, cpu
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    num_threads: Optional[int] = None


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    output_dir: str = "checkpoints"
    save_total_limit: int = 3
    save_best_only: bool = False
    metric_for_best: str = "loss"
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: str = "logs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        # Use OmegaConf for flexible config handling
        omega_conf = OmegaConf.create(config_dict)

        # Extract each section
        model_conf = ModelConfig(**omega_conf.get("model", {}))
        training_conf = TrainingConfig(**omega_conf.get("training", {}))
        data_conf = DataConfig(**omega_conf.get("data", {}))
        system_conf = SystemConfig(**omega_conf.get("system", {}))
        checkpoint_conf = CheckpointConfig(**omega_conf.get("checkpoint", {}))
        logging_conf = LoggingConfig(**omega_conf.get("logging", {}))

        return cls(
            model=model_conf,
            training=training_conf,
            data=data_conf,
            system=system_conf,
            checkpoint=checkpoint_conf,
            logging=logging_conf
        )

    def to_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "system": asdict(self.system),
            "checkpoint": asdict(self.checkpoint),
            "logging": asdict(self.logging)
        }

    def validate(self) -> bool:
        """Validate configuration"""
        errors = []

        # Validate model config
        if self.model.hidden_size <= 0:
            errors.append("model.hidden_size must be positive")
        if self.model.num_layers <= 0:
            errors.append("model.num_layers must be positive")
        if self.model.num_heads <= 0:
            errors.append("model.num_heads must be positive")
        if self.model.hidden_size % self.model.num_heads != 0:
            errors.append("model.hidden_size must be divisible by model.num_heads")

        # Validate training config
        if self.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")
        if self.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")
        if self.training.epochs <= 0:
            errors.append("training.epochs must be positive")

        # Validate data paths
        if not self.data.train_path:
            errors.append("data.train_path is required")

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True

    def __repr__(self) -> str:
        return f"Config(\n" + "\n".join(
            f"  {k}: {v}" for k, v in self.to_dict().items()
        ) + "\n)"
