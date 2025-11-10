# Training Guide

Complete guide to training models with IceNet AI.

## Table of Contents

1. [Configuration](#configuration)
2. [Data Preparation](#data-preparation)
3. [Training Process](#training-process)
4. [Monitoring](#monitoring)
5. [Optimization Tips](#optimization-tips)

## Configuration

### Model Configuration

```yaml
model:
  type: transformer  # transformer, cnn, lstm, gru
  hidden_size: 512  # Hidden dimension
  num_layers: 6  # Number of layers
  num_heads: 8  # Attention heads (transformer only)
  dropout: 0.1  # Dropout rate
  vocab_size: 50000  # Vocabulary size
  max_seq_length: 512  # Maximum sequence length
  activation: gelu  # relu, gelu, silu
  use_mixed_precision: true  # Enable FP16
  compile_model: false  # PyTorch 2.0 compilation
```

### Training Configuration

```yaml
training:
  batch_size: 32  # Batch size per device
  learning_rate: 0.0001  # Learning rate
  weight_decay: 0.01  # Weight decay (L2 regularization)
  epochs: 10  # Number of training epochs
  warmup_steps: 1000  # Learning rate warmup
  gradient_clip: 1.0  # Gradient clipping value
  optimizer: adamw  # adam, adamw, sgd
  scheduler: cosine  # linear, cosine, constant
  accumulation_steps: 1  # Gradient accumulation
  eval_steps: 500  # Evaluation frequency
  save_steps: 1000  # Checkpoint save frequency
  logging_steps: 100  # Logging frequency
  mixed_precision: fp16  # fp16, bf16, fp32
  gradient_checkpointing: false  # Memory optimization
```

### Data Configuration

```yaml
data:
  train_path: data/train  # Training data path
  val_path: data/val  # Validation data path
  test_path: data/test  # Test data path (optional)
  num_workers: 4  # DataLoader workers
  prefetch_factor: 2  # Prefetch batches
  pin_memory: true  # Pin memory for faster transfer
  shuffle: true  # Shuffle training data
  max_samples: null  # Limit samples (null = all)
```

## Data Preparation

### Text Data

Create `.txt` or `.jsonl` files:

**train.txt:**
```
First training example text
Second training example text
Third training example text
```

**train.jsonl:**
```json
{"text": "First training example"}
{"text": "Second training example"}
{"text": "Third training example"}
```

### Image Data

Organize images in directories:
```
data/
  train/
    class1/
      img1.jpg
      img2.jpg
    class2/
      img1.jpg
      img2.jpg
  val/
    class1/
      img1.jpg
    class2/
      img1.jpg
```

### Custom Dataset

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_your_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "labels": self.labels[idx]
        }
```

## Training Process

### Command Line

```bash
# Basic training
icenet train -c config.yaml

# Override config values
icenet train -c config.yaml --batch-size 64 --epochs 20

# Resume from checkpoint
icenet train -c config.yaml --resume checkpoint.pt
```

### Python Script

```python
from icenet import IceNetEngine, Config, Trainer
from torch.utils.data import DataLoader

# Load config
config = Config.from_yaml("config.yaml")

# Create engine
engine = IceNetEngine(config)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create trainer
trainer = Trainer(engine, train_loader, val_loader)

# Train
trainer.train()
```

### Interactive Mode

```bash
icenet  # Launch UI
# Navigate to Train tab (press 't')
# Enter config path
# Click "Start Training"
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/
```

View at: http://localhost:6006

### Weights & Biases

Enable in config:
```yaml
logging:
  use_wandb: true
  wandb_project: my-project
  wandb_run_name: experiment-1
```

### Real-time Terminal

Use interactive mode for live monitoring with beautiful visualizations.

## Optimization Tips

### Memory Optimization

For large models or limited memory:

```yaml
training:
  batch_size: 8  # Reduce batch size
  gradient_checkpointing: true  # Enable checkpointing
  mixed_precision: fp16  # Use FP16
  accumulation_steps: 4  # Maintain effective batch size
```

### Speed Optimization

For faster training:

```yaml
model:
  compile_model: true  # PyTorch 2.0 compilation

training:
  mixed_precision: fp16  # Faster than FP32

data:
  num_workers: 8  # More workers
  prefetch_factor: 4  # More prefetch

system:
  benchmark: true  # cuDNN benchmark
```

### Quality Optimization

For better model quality:

```yaml
training:
  batch_size: 64  # Larger batch
  learning_rate: 0.00005  # Lower LR
  warmup_steps: 2000  # Longer warmup
  epochs: 50  # More epochs
  gradient_clip: 0.5  # Tighter clipping
```

### Apple M4 Pro Specific

Optimize for M4 Pro unified memory:

```yaml
system:
  device: auto  # Auto-select MPS

training:
  batch_size: 32  # Adjust based on memory
  mixed_precision: fp16  # MPS supports FP16

data:
  num_workers: 4  # Good for M4 Pro CPU cores
  pin_memory: true  # Faster with unified memory
```

## Advanced Features

### Custom Callbacks

```python
from icenet.training.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        print(f"Epoch {epoch}: Loss = {train_metrics['loss']}")

trainer = Trainer(engine, train_loader, val_loader,
                 callbacks=[CustomCallback()])
```

### Learning Rate Finder

```python
# TODO: Implement LR finder
```

### Mixed Precision Training

Automatic mixed precision is enabled by default on Apple Silicon:

```yaml
training:
  mixed_precision: fp16  # or bf16
```

## Troubleshooting

### NaN Loss

- Reduce learning rate
- Enable gradient clipping
- Check data for NaN/Inf values
- Use mixed precision carefully

### Slow Training

- Increase batch size
- Enable compilation
- Use more data workers
- Check device utilization

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Reduce model size

## Best Practices

1. **Start Small**: Test with small model first
2. **Monitor Closely**: Watch loss curves for issues
3. **Save Often**: Use checkpointing
4. **Validate Regularly**: Check val loss frequently
5. **Document**: Keep notes on experiments

## Examples

See [examples/](../examples/) for complete training scripts.

## Next Steps

- Explore [Model Architectures](architecture.md)
- Learn about [Data Loading](data.md)
- Read the [API Reference](api.md)
