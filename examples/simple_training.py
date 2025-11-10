"""
Simple training example for IceNet AI

This example shows how to train a small transformer model
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from icenet import IceNetEngine, Config, Trainer
from icenet.data import TextDataset, SimpleTokenizer


def main():
    # Configuration
    config = Config.from_yaml("configs/transformer_small.yaml")

    # Create sample data (replace with your actual data)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Apple M4 Pro is powerful for AI",
        # Add more training data here
    ]

    # Build tokenizer
    tokenizer = SimpleTokenizer(mode="word")
    tokenizer.build_vocab(sample_texts)

    # Create datasets
    # For this example, we'll use the same data for train/val
    # In practice, split your data properly
    train_dataset = TextDataset(
        data_path="data/train.txt",  # Create this file with your data
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length,
    )

    val_dataset = TextDataset(
        data_path="data/val.txt",  # Create this file with your data
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Create engine
    print("Initializing IceNet engine...")
    engine = IceNetEngine(config)
    print(f"Engine: {engine}")

    # Create trainer
    print("\nStarting training...")
    trainer = Trainer(
        engine=engine,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Train
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
