# IceNet AI

A powerful, easy-to-use AI system optimized for Apple M4 Pro with a beautiful terminal interface.

## Features

- **M4 Pro Optimized**: Leverages Metal Performance Shaders (MPS) for GPU acceleration
- **Easy Training**: Simple configuration-based training system
- **Beautiful Terminal UI**: Nomadnet-inspired interface for intuitive interaction
- **Modular Architecture**: Support for multiple model types (Transformers, CNNs, RNNs)
- **Auto-Updates**: Built-in update checker and installer
- **Lightweight**: Efficient memory usage optimized for Apple Silicon

## System Requirements

- macOS 14.0 or later
- Apple Silicon (M1, M2, M3, M4 series)
- Python 3.10 or later
- 16GB+ RAM recommended

## Quick Installation

```bash
curl -sSL https://raw.githubusercontent.com/IceNet-01/IceNet-AI/main/install.sh | bash
```

Or manual installation:

```bash
git clone https://github.com/IceNet-01/IceNet-AI.git
cd IceNet-AI
./install.sh
```

## Quick Start

1. **Launch IceNet**:
   ```bash
   icenet
   ```

2. **Train a model**:
   ```bash
   icenet train --config configs/example.yaml
   ```

3. **Interactive mode**:
   ```bash
   icenet interactive
   ```

## Architecture

IceNet supports multiple neural network architectures:

- **Transformers**: State-of-the-art attention-based models
- **CNNs**: Convolutional networks for vision tasks
- **RNNs/LSTMs**: Recurrent networks for sequence modeling
- **Hybrid Models**: Custom architectures combining multiple approaches

All models are optimized for Apple Silicon using:
- Metal Performance Shaders (MPS)
- Unified Memory Architecture
- Neural Engine integration
- Mixed precision training (FP16/BF16)

## Training

Training is as simple as creating a YAML config:

```yaml
model:
  type: transformer
  hidden_size: 512
  num_layers: 6
  num_heads: 8

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10
  optimizer: adamw

data:
  train_path: data/train
  val_path: data/val
```

Then run:
```bash
icenet train --config my_config.yaml
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Training Guide](docs/training.md)
- [Model Architecture](docs/architecture.md)
- [API Reference](docs/api.md)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
