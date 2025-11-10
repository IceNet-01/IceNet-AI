"""Model architectures for IceNet"""

from icenet.models.transformer import TransformerModel
from icenet.models.cnn import CNNModel
from icenet.models.rnn import RNNModel, LSTMModel, GRUModel
from icenet.core.config import ModelConfig
import torch.nn as nn


def get_model(config: ModelConfig) -> nn.Module:
    """
    Factory function to get model by type

    Args:
        config: Model configuration

    Returns:
        Model instance
    """
    model_type = config.type.lower()

    if model_type == "transformer":
        return TransformerModel(config)
    elif model_type == "cnn":
        return CNNModel(config)
    elif model_type in ["rnn", "lstm"]:
        return LSTMModel(config)
    elif model_type == "gru":
        return GRUModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "TransformerModel",
    "CNNModel",
    "RNNModel",
    "LSTMModel",
    "GRUModel",
    "get_model",
]
