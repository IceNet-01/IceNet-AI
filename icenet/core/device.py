"""
Device management optimized for Apple Silicon M4 Pro
"""

import torch
import platform
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Information about the device"""
    device: torch.device
    device_name: str
    has_mps: bool
    has_neural_engine: bool
    unified_memory_gb: float
    cpu_cores: int
    gpu_cores: int
    supports_fp16: bool
    supports_bf16: bool


class DeviceManager:
    """Manages device selection and optimization for Apple Silicon"""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize device manager

        Args:
            device: Device to use ('auto', 'mps', 'cpu'). If None, auto-detect.
        """
        self.device = self._setup_device(device)
        self.info = self._get_device_info()

    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """Setup and return the optimal device"""
        if device == "cpu":
            return torch.device("cpu")

        # Check for Apple Silicon MPS (Metal Performance Shaders)
        if torch.backends.mps.is_available():
            try:
                # Test MPS availability
                torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception as e:
                print(f"MPS available but not working: {e}")
                print("Falling back to CPU")
                return torch.device("cpu")

        # Fallback to CPU
        return torch.device("cpu")

    def _get_device_info(self) -> DeviceInfo:
        """Get detailed device information"""
        is_mac = platform.system() == "Darwin"
        has_mps = self.device.type == "mps"

        # Get system info for M4 Pro
        cpu_cores = self._get_cpu_cores()
        gpu_cores = self._get_gpu_cores()
        unified_memory = self._get_unified_memory()

        # Check for Neural Engine (present on all Apple Silicon)
        has_neural_engine = is_mac and platform.processor() == "arm"

        # M4 Pro supports both FP16 and BF16
        supports_fp16 = has_mps
        supports_bf16 = has_mps

        device_name = self._get_device_name()

        return DeviceInfo(
            device=self.device,
            device_name=device_name,
            has_mps=has_mps,
            has_neural_engine=has_neural_engine,
            unified_memory_gb=unified_memory,
            cpu_cores=cpu_cores,
            gpu_cores=gpu_cores,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16
        )

    def _get_device_name(self) -> str:
        """Get the device name"""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
        except:
            pass
        return platform.processor()

    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores"""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.ncpu"],
                    capture_output=True,
                    text=True
                )
                return int(result.stdout.strip())
        except:
            pass
        return torch.get_num_threads()

    def _get_gpu_cores(self) -> int:
        """Get number of GPU cores (M4 Pro typically has 16-20)"""
        try:
            if platform.system() == "Darwin":
                # For M4 Pro, estimate based on model
                # This is an approximation as direct query isn't available
                return 16  # M4 Pro typically has 16-20 GPU cores
        except:
            pass
        return 0

    def _get_unified_memory(self) -> float:
        """Get unified memory in GB"""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True
                )
                bytes_mem = int(result.stdout.strip())
                return bytes_mem / (1024 ** 3)  # Convert to GB
        except:
            pass
        return 0.0

    def get_optimal_batch_size(self, model_size_mb: float) -> int:
        """
        Calculate optimal batch size based on available memory

        Args:
            model_size_mb: Size of the model in MB

        Returns:
            Recommended batch size
        """
        available_memory_mb = self.info.unified_memory_gb * 1024 * 0.7  # Use 70% of RAM

        # Reserve memory for model and gradients (3x model size)
        available_for_batch = available_memory_mb - (model_size_mb * 3)

        # Estimate ~100MB per batch item (conservative)
        estimated_batch_size = int(available_for_batch / 100)

        # Clamp between reasonable values
        return max(1, min(estimated_batch_size, 128))

    def optimize_for_training(self):
        """Apply optimizations for training"""
        if self.info.has_mps:
            # Enable Metal optimizations
            torch.backends.mps.enable_fallback_to_cpu = True

        # Set number of threads for CPU operations
        torch.set_num_threads(max(1, self.info.cpu_cores - 2))

    def optimize_for_inference(self):
        """Apply optimizations for inference"""
        if self.info.has_mps:
            torch.backends.mps.enable_fallback_to_cpu = True

        torch.set_num_threads(self.info.cpu_cores)

    def get_info_dict(self) -> Dict[str, Any]:
        """Get device info as dictionary"""
        return {
            "device": str(self.info.device),
            "device_name": self.info.device_name,
            "has_mps": self.info.has_mps,
            "has_neural_engine": self.info.has_neural_engine,
            "unified_memory_gb": round(self.info.unified_memory_gb, 2),
            "cpu_cores": self.info.cpu_cores,
            "gpu_cores": self.info.gpu_cores,
            "supports_fp16": self.info.supports_fp16,
            "supports_bf16": self.info.supports_bf16,
        }

    def __repr__(self) -> str:
        info = self.get_info_dict()
        return (
            f"DeviceManager(\n"
            f"  Device: {info['device']}\n"
            f"  Name: {info['device_name']}\n"
            f"  MPS: {info['has_mps']}\n"
            f"  Neural Engine: {info['has_neural_engine']}\n"
            f"  Memory: {info['unified_memory_gb']} GB\n"
            f"  CPU Cores: {info['cpu_cores']}\n"
            f"  GPU Cores: {info['gpu_cores']}\n"
            f")"
        )
