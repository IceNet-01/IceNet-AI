"""RNN/LSTM/GRU model architectures optimized for Apple Silicon"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from icenet.models.base import BaseModel
from icenet.core.config import ModelConfig


class RNNModel(BaseModel):
    """
    Base RNN model

    Supports LSTM, GRU, and vanilla RNN
    """

    def __init__(self, config: ModelConfig, rnn_type: str = "lstm"):
        super().__init__(config)

        self.rnn_type = rnn_type.lower()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # RNN layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=False,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                self.hidden_size,
                self.hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=False,
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                self.hidden_size,
                self.hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=False,
                nonlinearity="relu",
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        # Output layer
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        # Initialize output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        return_hidden: bool = False,
    ):
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            hidden: Hidden state from previous step (optional)
            return_hidden: If True, return hidden states

        Returns:
            logits and optionally hidden states
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        embedded = self.dropout_layer(embedded)

        # RNN forward
        if hidden is not None:
            output, hidden_state = self.rnn(embedded, hidden)
        else:
            output, hidden_state = self.rnn(embedded)

        # Output projection
        output = self.dropout_layer(output)
        logits = self.fc(output)  # [batch_size, seq_len, vocab_size]

        if return_hidden:
            return logits, hidden_state

        return logits

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state"""
        if self.rnn_type == "lstm":
            h = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
            c = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
            return (h, c)
        else:
            return torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token IDs
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize hidden state
        hidden = self.init_hidden(batch_size, device)

        # Process input sequence
        with torch.no_grad():
            _, hidden = self.forward(input_ids, hidden, return_hidden=True)

            # Generate tokens one by one
            current_token = input_ids[:, -1:]  # Last token

            generated = input_ids

            for _ in range(max_length):
                logits, hidden = self.forward(current_token, hidden, return_hidden=True)
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                current_token = next_token

        return generated


class LSTMModel(RNNModel):
    """LSTM model"""

    def __init__(self, config: ModelConfig):
        super().__init__(config, rnn_type="lstm")


class GRUModel(RNNModel):
    """GRU model"""

    def __init__(self, config: ModelConfig):
        super().__init__(config, rnn_type="gru")
