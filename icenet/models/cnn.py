"""CNN model architecture optimized for Apple Silicon"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from icenet.models.base import BaseModel
from icenet.core.config import ModelConfig


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper CNNs"""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv1 = ConvBlock(channels, channels, 3, 1, 1, dropout, activation)
        self.conv2 = ConvBlock(channels, channels, 3, 1, 1, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class CNNModel(BaseModel):
    """
    CNN model optimized for Apple Silicon M4 Pro

    Supports both classification and feature extraction
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Configuration
        self.num_classes = config.vocab_size  # Reuse vocab_size as num_classes
        self.in_channels = 3  # RGB images by default
        self.dropout = config.dropout
        self.activation = config.activation

        # Calculate channel progression
        base_channels = 64
        channels = [base_channels * (2 ** i) for i in range(config.num_layers)]

        # Input convolution
        self.input_conv = ConvBlock(
            self.in_channels, base_channels, 7, 2, 3, self.dropout, self.activation
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Build layers
        layers = []
        in_ch = base_channels

        for i, out_ch in enumerate(channels):
            # Add convolutional block
            layers.append(
                ConvBlock(in_ch, out_ch, 3, 2 if i > 0 else 1, 1, self.dropout, self.activation)
            )

            # Add residual blocks for deeper networks
            if config.num_layers > 4:
                layers.append(ResidualBlock(out_ch, self.dropout, self.activation))

            in_ch = out_ch

        self.layers = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(channels[-1], config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(config.hidden_size, self.num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch_size, channels, height, width] - Input images
            return_features: If True, return features instead of logits

        Returns:
            logits or features
        """
        # Input convolution
        x = self.input_conv(x)
        x = self.maxpool(x)

        # Main layers
        x = self.layers(x)

        # Global pooling
        features = self.global_pool(x)
        features = features.view(features.size(0), -1)

        if return_features:
            return features

        # Classification
        logits = self.classifier(features)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input"""
        return self.forward(x, return_features=True)
