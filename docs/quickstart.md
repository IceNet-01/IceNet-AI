# IceNet AI - Quick Start Guide

Get started with IceNet AI in minutes!

## Installation

### Easy Install (macOS)

```bash
curl -sSL https://raw.githubusercontent.com/IceNet-01/IceNet-AI/main/install.sh | bash
```

### Manual Install

```bash
git clone https://github.com/IceNet-01/IceNet-AI.git
cd IceNet-AI
./install.sh
```

### Install via pip (when published)

```bash
pip install icenet-ai
```

## First Steps

### 1. Check Your System

```bash
icenet info
```

This displays your device information and confirms IceNet is optimized for your hardware.

### 2. Launch Interactive Mode

```bash
icenet
```

This opens the beautiful terminal UI where you can:
- Monitor training in real-time
- Manage models
- View system status

### 3. Train Your First Model

Create a simple config:

```bash
icenet config -o my_first_model.yaml
```

Start training:

```bash
icenet train -c my_first_model.yaml
```

## Understanding Configs

IceNet uses YAML configurations. Here's a minimal example:

```yaml
model:
  type: transformer
  hidden_size: 256
  num_layers: 4
  num_heads: 4

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10

data:
  train_path: data/train
  val_path: data/val
```

## Model Types

IceNet supports multiple architectures:

- **Transformer**: State-of-the-art for NLP
  ```yaml
  model:
    type: transformer
  ```

- **CNN**: Great for image tasks
  ```yaml
  model:
    type: cnn
  ```

- **LSTM/GRU**: Excellent for sequences
  ```yaml
  model:
    type: lstm
  ```

## Memory Optimization

For large models on M4 Pro:

```yaml
training:
  batch_size: 16  # Reduce if OOM
  mixed_precision: fp16  # Use FP16
  gradient_checkpointing: true  # Save memory
  accumulation_steps: 2  # Effective batch size = 32
```

## Quick Commands

```bash
# Launch UI
icenet

# Train model
icenet train -c config.yaml

# Evaluate model
icenet eval --checkpoint model.pt

# Generate text
icenet generate --checkpoint model.pt --prompt "Hello"

# Check for updates
icenet update

# Get help
icenet --help
```

## Next Steps

- Read the [Training Guide](training.md) for detailed training instructions
- Check out [Examples](../examples/) for code samples
- Explore [Model Architectures](architecture.md) to understand the models
- Join our community on GitHub

## Troubleshooting

### Out of Memory

Reduce batch size:
```yaml
training:
  batch_size: 16  # Try 8, 4, or even 1
```

Enable gradient checkpointing:
```yaml
training:
  gradient_checkpointing: true
```

### Slow Training

Check device:
```bash
icenet info
```

Ensure MPS is enabled. If not, check macOS version (requires 14.0+).

### Import Errors

Reinstall:
```bash
pip install --upgrade --force-reinstall icenet-ai
```

## Getting Help

- Issues: https://github.com/IceNet-01/IceNet-AI/issues
- Discussions: https://github.com/IceNet-01/IceNet-AI/discussions
- Documentation: https://github.com/IceNet-01/IceNet-AI/docs

Happy training! 🚀
