"""
IceNet AI - A powerful AI system optimized for Apple M4 Pro
"""

__version__ = "0.1.0"
__author__ = "IceNet AI Team"

from icenet.core.engine import IceNetEngine
from icenet.core.config import Config
from icenet.training.trainer import Trainer

__all__ = ["IceNetEngine", "Config", "Trainer", "__version__"]
