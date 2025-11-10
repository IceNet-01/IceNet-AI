"""
Main IceNet AI Engine
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

from icenet.core.config import Config
from icenet.core.device import DeviceManager
from icenet.models import get_model


logger = logging.getLogger(__name__)


class IceNetEngine:
    """Main engine for IceNet AI system"""

    def __init__(
        self,
        config: Optional[Union[Config, str, Path]] = None,
        model: Optional[nn.Module] = None,
    ):
        """
        Initialize IceNet Engine

        Args:
            config: Configuration object or path to config file
            model: Pre-initialized model (optional)
        """
        # Load or create config
        if config is None:
            self.config = Config()
        elif isinstance(config, (str, Path)):
            self.config = Config.from_yaml(config)
        else:
            self.config = config

        # Validate config
        self.config.validate()

        # Setup device
        self.device_manager = DeviceManager(self.config.system.device)
        self.device = self.device_manager.device

        logger.info(f"Initialized IceNet Engine on {self.device}")
        logger.info(f"Device info: {self.device_manager}")

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._build_model()

        # Track model state
        self.is_training = False
        self._compiled = False

    def _build_model(self) -> nn.Module:
        """Build model from configuration"""
        logger.info(f"Building {self.config.model.type} model...")

        model = get_model(self.config.model)
        model = model.to(self.device)

        # Compile model if enabled (PyTorch 2.0+)
        if self.config.model.compile_model:
            try:
                logger.info("Compiling model with torch.compile...")
                model = torch.compile(model)
                self._compiled = True
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model built successfully:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        return model

    def train_mode(self):
        """Set engine to training mode"""
        self.model.train()
        self.is_training = True
        self.device_manager.optimize_for_training()

    def eval_mode(self):
        """Set engine to evaluation mode"""
        self.model.eval()
        self.is_training = False
        self.device_manager.optimize_for_inference()

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the model"""
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Make engine callable"""
        return self.forward(*args, **kwargs)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch (optional)
            global_step: Current global step (optional)
            metadata: Additional metadata (optional)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metadata is not None:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            strict: Whether to strictly enforce state dict keys match

        Returns:
            Dictionary containing checkpoint metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from {path}")

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded")

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state loaded")

        return {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "metadata": checkpoint.get("metadata", {}),
        }

    def get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for current model and device"""
        model_size_mb = self.get_model_size()
        return self.device_manager.get_optimal_batch_size(model_size_mb)

    def summary(self) -> Dict[str, Any]:
        """Get engine summary"""
        return {
            "device": str(self.device),
            "device_info": self.device_manager.get_info_dict(),
            "model_type": self.config.model.type,
            "model_size_mb": round(self.get_model_size(), 2),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "is_training": self.is_training,
            "is_compiled": self._compiled,
        }

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            f"IceNetEngine(\n"
            f"  Device: {summary['device']}\n"
            f"  Model: {summary['model_type']}\n"
            f"  Parameters: {summary['total_parameters']:,}\n"
            f"  Size: {summary['model_size_mb']:.2f} MB\n"
            f"  Training: {summary['is_training']}\n"
            f")"
        )
