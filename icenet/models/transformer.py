"""Transformer model architecture optimized for Apple Silicon"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from icenet.models.base import BaseModel
from icenet.core.config import ModelConfig


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: Optional mask
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = self.out_linear(context)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TransformerModel(BaseModel):
    """
    Transformer model optimized for Apple Silicon M4 Pro

    Supports both encoder-only and encoder-decoder architectures
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.d_model = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.d_ff = config.hidden_size * 4  # Standard transformer uses 4x

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(
            self.d_model, config.max_seq_length, config.dropout
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.d_model,
                self.num_heads,
                self.d_ff,
                config.dropout,
                config.activation
            )
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(self.d_model, config.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len] - Token IDs
            attention_mask: Optional [batch_size, seq_len] - Attention mask
            return_embeddings: If True, return embeddings instead of logits

        Returns:
            logits: [batch_size, seq_len, vocab_size] or embeddings
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        if return_embeddings:
            return x

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token IDs
        """
        self.eval()
        batch_size = input_ids.size(0)

        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits = self.forward(input_ids)  # [batch_size, seq_len, vocab_size]

                # Get last token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
