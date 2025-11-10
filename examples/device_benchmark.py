"""
Device benchmark for Apple M4 Pro

This script benchmarks IceNet on your device
"""

import torch
import time
from icenet.core.device import DeviceManager
from icenet.core.config import Config, ModelConfig
from icenet.models import get_model


def benchmark_inference(model, device, batch_size=32, seq_len=512, num_runs=100):
    """Benchmark inference speed"""
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(
        0, 1000, (batch_size, seq_len), device=device
    )

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device.type == "mps":
        # MPS doesn't have synchronize, use a different method
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    elapsed = end_time - start_time
    throughput = (num_runs * batch_size) / elapsed

    return throughput, elapsed


def main():
    print("=" * 60)
    print("IceNet Device Benchmark")
    print("=" * 60)
    print()

    # Get device info
    device_manager = DeviceManager()
    print("Device Information:")
    print(device_manager)
    print()

    # Test different model sizes
    model_configs = [
        ("Small", ModelConfig(hidden_size=256, num_layers=4, num_heads=4)),
        ("Medium", ModelConfig(hidden_size=512, num_layers=6, num_heads=8)),
        ("Large", ModelConfig(hidden_size=1024, num_layers=8, num_heads=16)),
    ]

    print("Benchmarking inference speed...")
    print("-" * 60)

    for name, config in model_configs:
        print(f"\nModel: {name}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Heads: {config.num_heads}")

        # Create model
        model = get_model(config)
        model = model.to(device_manager.device)

        # Benchmark
        throughput, elapsed = benchmark_inference(
            model, device_manager.device, batch_size=16, seq_len=256, num_runs=50
        )

        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Time: {elapsed:.3f} seconds")

        # Get model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = param_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")

        del model

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
