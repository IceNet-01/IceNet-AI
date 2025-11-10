"""
Inference example for IceNet AI

This example shows how to load a trained model and generate text
"""

import torch
from icenet import IceNetEngine, Config
from icenet.data import SimpleTokenizer


def main():
    # Load config and model
    config_path = "configs/transformer_small.yaml"
    checkpoint_path = "checkpoints/transformer_small/checkpoint_best.pt"

    print("Loading model...")
    config = Config.from_yaml(config_path)
    engine = IceNetEngine(config)

    # Load checkpoint
    engine.load_checkpoint(checkpoint_path)
    engine.eval_mode()

    print("Model loaded successfully!")
    print(f"\nModel summary:")
    for key, value in engine.summary().items():
        print(f"  {key}: {value}")

    # Load tokenizer (you should save/load your trained tokenizer)
    tokenizer = SimpleTokenizer(mode="word")
    # tokenizer = SimpleTokenizer.load("tokenizer.json")

    # Generate text
    prompt = "The quick brown fox"
    print(f"\nPrompt: {prompt}")

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids]).to(engine.device)

    # Generate
    with torch.no_grad():
        output_ids = engine.model.generate(
            input_ids,
            max_length=50,
            temperature=0.8,
            top_k=50,
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
