"""Simple tokenizer for IceNet"""

from typing import List, Dict
from pathlib import Path
import json
import re


class SimpleTokenizer:
    """Simple character/word-level tokenizer"""

    def __init__(
        self,
        vocab: Dict[str, int] = None,
        mode: str = "char",  # char or word
        vocab_size: int = 10000,
    ):
        """
        Initialize tokenizer

        Args:
            vocab: Pre-built vocabulary
            mode: Tokenization mode ('char' or 'word')
            vocab_size: Maximum vocabulary size
        """
        self.mode = mode
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Vocabulary
        if vocab is not None:
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in vocab.items()}
        else:
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3,
            }
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        token_counts = {}

        for text in texts:
            tokens = self._tokenize_text(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_tokens = sorted(
            token_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Build vocab
        for token, _ in sorted_tokens[: self.vocab_size - len(self.vocab)]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Update inverse vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into tokens"""
        if self.mode == "char":
            return list(text)
        elif self.mode == "word":
            # Simple word tokenization
            text = text.lower()
            tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
            return tokens
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = self._tokenize_text(text)

        # Convert to IDs
        ids = []

        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])

        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab[self.unk_token]))

        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokens = []

        for id in ids:
            token = self.inv_vocab.get(id, self.unk_token)

            if skip_special_tokens and token in [
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token,
            ]:
                continue

            tokens.append(token)

        if self.mode == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)

    def save(self, path: str):
        """Save tokenizer to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab": self.vocab,
            "mode": self.mode,
            "vocab_size": self.vocab_size,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer from file"""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            vocab=data["vocab"],
            mode=data["mode"],
            vocab_size=data["vocab_size"],
        )

    def __call__(self, text: str) -> List[int]:
        """Make tokenizer callable"""
        return self.encode(text)

    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)
