# Model Architecture Guide

Deep dive into IceNet's model architectures.

## Overview

IceNet supports three main architecture families:

1. **Transformers** - Attention-based models for NLP
2. **CNNs** - Convolutional networks for vision
3. **RNNs** - Recurrent networks for sequences (LSTM/GRU)

All architectures are optimized for Apple Silicon M4 Pro.

## Transformer Architecture

### Overview

IceNet's transformer implements the standard architecture with optimizations:

```
Input → Embedding → Positional Encoding →
  N × [Multi-Head Attention → Feed-Forward] →
  Output Projection
```

### Configuration

```yaml
model:
  type: transformer
  hidden_size: 512        # Model dimension (d_model)
  num_layers: 6           # Number of transformer blocks
  num_heads: 8            # Attention heads
  dropout: 0.1            # Dropout rate
  vocab_size: 50000       # Vocabulary size
  max_seq_length: 512     # Maximum sequence length
  activation: gelu        # Activation function
```

### Components

**Multi-Head Attention:**
- Scaled dot-product attention
- Multiple attention heads for diverse representations
- Optimized for MPS backend

**Feed-Forward Network:**
- Two-layer MLP with expansion factor 4x
- GELU activation (default)
- Dropout for regularization

**Positional Encoding:**
- Sinusoidal positional embeddings
- Added to input embeddings
- Supports sequences up to max_seq_length

### Memory Requirements

Approximate memory usage:

| Model Size | Hidden | Layers | Heads | Parameters | Memory (FP16) |
|------------|--------|--------|-------|------------|---------------|
| Tiny       | 256    | 4      | 4     | ~20M       | ~40 MB        |
| Small      | 512    | 6      | 8     | ~90M       | ~180 MB       |
| Medium     | 768    | 12     | 12    | ~250M      | ~500 MB       |
| Large      | 1024   | 12     | 16    | ~450M      | ~900 MB       |
| XL         | 1536   | 24     | 24    | ~1.5B      | ~3 GB         |

### Usage

```python
from icenet.models import TransformerModel
from icenet.core.config import ModelConfig

config = ModelConfig(
    type="transformer",
    hidden_size=512,
    num_layers=6,
    num_heads=8,
)

model = TransformerModel(config)

# Forward pass
output = model(input_ids)  # [batch, seq_len, vocab_size]

# Generation
generated = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
)
```

## CNN Architecture

### Overview

Convolutional Neural Network for image tasks:

```
Input → Conv Block → N × [Conv/Residual Blocks] →
  Global Pooling → Classifier
```

### Configuration

```yaml
model:
  type: cnn
  hidden_size: 512        # Classifier hidden size
  num_layers: 5           # Number of conv layers
  dropout: 0.2            # Dropout rate
  vocab_size: 1000        # Number of classes
  activation: relu        # Activation function
```

### Components

**Conv Block:**
- Convolution → Batch Norm → Activation → Dropout
- Progressive channel increase
- Stride-2 for downsampling

**Residual Block:**
- Skip connections for deeper networks
- Improves gradient flow
- Better training stability

**Global Average Pooling:**
- Reduces spatial dimensions to 1×1
- Fewer parameters than FC layers
- Translation invariance

### Memory Requirements

| Model Size | Base Ch | Layers | Parameters | Memory (FP16) |
|------------|---------|--------|------------|---------------|
| Small      | 64      | 4      | ~5M        | ~10 MB        |
| Medium     | 128     | 6      | ~20M       | ~40 MB        |
| Large      | 256     | 8      | ~80M       | ~160 MB       |

### Usage

```python
from icenet.models import CNNModel
from icenet.core.config import ModelConfig

config = ModelConfig(
    type="cnn",
    hidden_size=512,
    num_layers=5,
    vocab_size=1000,  # num_classes
)

model = CNNModel(config)

# Forward pass
logits = model(images)  # [batch, num_classes]

# Feature extraction
features = model.extract_features(images)  # [batch, hidden_size]
```

## RNN Architecture

### Overview

Recurrent networks for sequence modeling:

```
Input → Embedding → N × RNN Layers → Output Projection
```

Supports LSTM, GRU, and vanilla RNN.

### Configuration

```yaml
model:
  type: lstm              # lstm, gru, or rnn
  hidden_size: 512        # Hidden state size
  num_layers: 4           # Number of RNN layers
  dropout: 0.3            # Dropout between layers
  vocab_size: 10000       # Vocabulary size
```

### Components

**LSTM Cell:**
- Input, forget, output gates
- Cell state for long-term memory
- Better gradient flow than vanilla RNN

**GRU Cell:**
- Simplified LSTM with reset/update gates
- Fewer parameters
- Often similar performance

**Embedding:**
- Learnable token embeddings
- Input to RNN layers

### Memory Requirements

| Model Size | Hidden | Layers | Parameters | Memory (FP16) |
|------------|--------|--------|------------|---------------|
| Small      | 256    | 2      | ~5M        | ~10 MB        |
| Medium     | 512    | 4      | ~20M       | ~40 MB        |
| Large      | 1024   | 4      | ~80M       | ~160 MB       |

### Usage

```python
from icenet.models import LSTMModel, GRUModel
from icenet.core.config import ModelConfig

config = ModelConfig(
    type="lstm",
    hidden_size=512,
    num_layers=4,
)

model = LSTMModel(config)

# Forward pass
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# With hidden state
logits, hidden = model(input_ids, hidden, return_hidden=True)

# Generation
generated = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
)
```

## Optimization for Apple Silicon

### Metal Performance Shaders (MPS)

All models automatically use MPS when available:

- GPU acceleration on Apple Silicon
- Optimized kernels for common operations
- Unified memory architecture benefits

### Mixed Precision

FP16 training for 2x memory savings:

```python
# Automatic in training pipeline
config.training.mixed_precision = "fp16"
```

### Model Compilation

PyTorch 2.0 compilation for faster execution:

```python
config.model.compile_model = True
```

### Gradient Checkpointing

Trade compute for memory:

```python
config.training.gradient_checkpointing = True
```

## Custom Architectures

Extend base model class:

```python
from icenet.models.base import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Define layers
        self.custom_layer = nn.Linear(
            config.hidden_size,
            config.vocab_size
        )

    def forward(self, x):
        # Define forward pass
        return self.custom_layer(x)
```

## Performance Tips

1. **Batch Size**: Use largest batch that fits in memory
2. **Sequence Length**: Shorter = faster (transformer)
3. **Model Width**: More efficient than depth on M4 Pro
4. **Attention**: Most expensive in transformers
5. **Mixed Precision**: Always use FP16 on Apple Silicon

## Benchmarks

On Apple M4 Pro (24GB):

| Model       | Size   | Batch | Throughput    |
|-------------|--------|-------|---------------|
| Transformer | Small  | 32    | ~500 tok/s    |
| Transformer | Medium | 16    | ~300 tok/s    |
| Transformer | Large  | 8     | ~150 tok/s    |
| CNN         | Medium | 64    | ~1000 img/s   |
| LSTM        | Medium | 64    | ~800 tok/s    |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)

## Next Steps

- Try different architectures for your task
- Experiment with model sizes
- Read [Training Guide](training.md) for optimization
