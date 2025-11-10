"""Base model class"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from icenet.core.config import ModelConfig


class BaseModel(nn.Module, ABC):
    """Base class for all IceNet models"""

    def __init__(self, config: ModelConfig):
        """
        Initialize base model

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass - must be implemented by subclasses"""
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def summary(self) -> Dict[str, Any]:
        """Get model summary"""
        return {
            "type": self.__class__.__name__,
            "total_parameters": self.get_num_parameters(),
            "trainable_parameters": self.get_trainable_parameters(),
            "config": self.config.__dict__,
        }
