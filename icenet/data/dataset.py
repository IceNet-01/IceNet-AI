"""Dataset classes for IceNet"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Callable
import json
from PIL import Image
import numpy as np


class TextDataset(Dataset):
    """Dataset for text data"""

    def __init__(
        self,
        data_path: str,
        tokenizer: Callable,
        max_length: int = 512,
        return_labels: bool = True,
    ):
        """
        Initialize text dataset

        Args:
            data_path: Path to text data (txt or jsonl file)
            tokenizer: Tokenizer function
            max_length: Maximum sequence length
            return_labels: Whether to return labels (for language modeling)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_labels = return_labels

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> List[str]:
        """Load text data from file"""
        data = []

        if self.data_path.suffix == ".txt":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = [line.strip() for line in f if line.strip()]

        elif self.data_path.suffix == ".jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if "text" in item:
                        data.append(item["text"])

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]

        # Tokenize
        tokens = self.tokenizer(text)

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)

        result = {"input_ids": input_ids}

        if self.return_labels:
            # For language modeling, labels are shifted input_ids
            result["labels"] = input_ids.clone()

        return result


class ImageDataset(Dataset):
    """Dataset for image data"""

    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        """
        Initialize image dataset

        Args:
            data_path: Path to image directory or file list
            transform: Optional transform to apply
            image_size: Target image size
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.image_size = image_size

        # Load image paths
        self.image_paths = self._load_image_paths()

        # Default transform if none provided
        if self.transform is None:
            self.transform = self._default_transform()

    def _load_image_paths(self) -> List[Path]:
        """Load image paths"""
        if self.data_path.is_dir():
            # Load all images from directory
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            paths = []
            for ext in extensions:
                paths.extend(self.data_path.glob(f"*{ext}"))
                paths.extend(self.data_path.glob(f"*{ext.upper()}"))
            return sorted(paths)
        else:
            # Load from file list
            with open(self.data_path, "r") as f:
                return [Path(line.strip()) for line in f if line.strip()]

    def _default_transform(self):
        """Default image transform"""
        try:
            from torchvision import transforms

            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        except ImportError:
            # Fallback if torchvision not available
            def simple_transform(img):
                img = img.resize((self.image_size, self.image_size))
                img_array = np.array(img).astype(np.float32) / 255.0
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                return torch.from_numpy(img_array).permute(2, 0, 1)

            return simple_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transform
        image = self.transform(image)

        # Dummy label (can be extended for classification)
        label = 0

        return {
            "input_ids": image,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SequenceDataset(Dataset):
    """Generic sequence dataset"""

    def __init__(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
    ):
        """
        Initialize sequence dataset

        Args:
            sequences: List of token sequences
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.max_length = max_length

        # Determine max length if not provided
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]

        # Pad or truncate
        if len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))

        input_ids = torch.tensor(sequence, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }
