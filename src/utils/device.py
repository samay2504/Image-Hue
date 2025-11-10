"""
Unified device detection and configuration for GPU training.

Provides utilities for:
- Auto-detection of CUDA devices
- Safe batch size and num_workers recommendations
- CUDA environment setup
- Performance tuning for RTX 5070 Ti (12GB VRAM)
"""

import os
import torch
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_device(prefer: str = "cuda") -> torch.device:
    """
    Get the best available device for training/inference.
    
    Args:
        prefer: Preferred device type ("cuda" or "cpu")
    
    Returns:
        torch.device instance
    """
    # Check environment override
    env_device = os.environ.get("CUDA_DEVICE", os.environ.get("DEVICE", None))
    if env_device:
        prefer = env_device
    
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        if prefer == "cuda":
            logger.warning("CUDA requested but not available, falling back to CPU")
        else:
            logger.info("Using CPU device")
    
    return device


def get_device_info() -> dict:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device details
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "capability": f"{props.major}.{props.minor}",
            }
    
    return info


def auto_batch_and_workers(
    model_mem_estimate: Optional[int] = None,
    image_size: int = 224,
    vram_gb: Optional[float] = None
) -> Tuple[int, int]:
    """
    Automatically determine safe batch size and num_workers.
    
    Args:
        model_mem_estimate: Estimated model memory in MB (optional)
        image_size: Input image size
        vram_gb: GPU VRAM in GB (auto-detected if None)
    
    Returns:
        Tuple of (batch_size, num_workers)
    """
    # Detect VRAM
    if vram_gb is None and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif vram_gb is None:
        vram_gb = 0  # CPU mode
    
    # Determine batch size based on VRAM and model size
    if not torch.cuda.is_available():
        # CPU mode: smaller batches
        batch_size = 8
    elif vram_gb >= 12:  # RTX 5070 Ti or better
        if model_mem_estimate and model_mem_estimate > 2000:  # Large model (>2GB)
            batch_size = 8
        elif image_size >= 512:
            batch_size = 16
        elif image_size >= 256:
            batch_size = 32
        else:
            batch_size = 64
    elif vram_gb >= 8:  # Mid-range GPUs
        if image_size >= 512:
            batch_size = 8
        elif image_size >= 256:
            batch_size = 16
        else:
            batch_size = 32
    elif vram_gb >= 4:  # Entry-level GPUs
        if image_size >= 512:
            batch_size = 4
        else:
            batch_size = 8
    else:
        batch_size = 4
    
    # Determine num_workers based on CPU cores
    cpu_count = os.cpu_count() or 1
    
    if cpu_count >= 16:  # High-end systems (like Ryzen 7 9700X with 16 threads)
        num_workers = 12  # 75% of cores, good for 16-thread CPUs
    elif cpu_count >= 8:
        num_workers = 6
    elif cpu_count >= 4:
        num_workers = 4
    else:
        num_workers = 2
    
    # Reduce workers if batch size is very small (not worth the overhead)
    if batch_size <= 4:
        num_workers = min(num_workers, 4)
    
    logger.info(f"Auto-configured: batch_size={batch_size}, num_workers={num_workers}")
    logger.info(f"  VRAM: {vram_gb:.1f} GB, CPU cores: {cpu_count}, Image size: {image_size}")
    
    return batch_size, num_workers


def set_cuda_env(cuda_visible_devices: Optional[str] = None):
    """
    Set CUDA environment variables.
    
    Args:
        cuda_visible_devices: Comma-separated GPU indices (e.g., "0" or "0,1")
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={cuda_visible_devices}")


def enable_cuda_optimizations():
    """
    Enable CUDA optimizations for training performance.
    
    Call this at the start of training scripts for RTX 5070 Ti.
    """
    if torch.cuda.is_available():
        # Enable cuDNN benchmark mode for faster convolutions
        # Good for fixed input sizes (typical in training)
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode")
        
        # Ensure cuDNN is enabled
        if not torch.backends.cudnn.enabled:
            torch.backends.cudnn.enabled = True
            logger.info("Enabled cuDNN")
        
        # For deterministic results (disable if you need reproducibility)
        # torch.backends.cudnn.deterministic = True
        
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")


def estimate_model_memory(model: torch.nn.Module) -> float:
    """
    Estimate model memory usage in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Estimated memory in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_mb = (param_size + buffer_size) / (1024 ** 2)
    
    logger.info(f"Model memory estimate: {total_mb:.2f} MB")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return total_mb


def get_dataloader_config(
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> dict:
    """
    Get optimized DataLoader configuration for RTX 5070 Ti + 64GB RAM system.
    
    Args:
        batch_size: Override auto-detected batch size
        num_workers: Override auto-detected num_workers
        pin_memory: Override pin_memory (default: True for CUDA)
    
    Returns:
        Dictionary of DataLoader kwargs
    """
    if batch_size is None or num_workers is None:
        auto_batch, auto_workers = auto_batch_and_workers()
        batch_size = batch_size or auto_batch
        num_workers = num_workers or auto_workers
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    config = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,  # Keep workers alive between epochs
        "prefetch_factor": 2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
    }
    
    logger.info(f"DataLoader config: {config}")
    
    return config


def print_device_summary():
    """Print a summary of device configuration."""
    print("\n" + "="*60)
    print("Device Configuration Summary")
    print("="*60)
    
    info = get_device_info()
    
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"GPU Count: {info['device_count']}")
        
        for i in range(info['device_count']):
            gpu_info = info[f'gpu_{i}']
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu_info['name']}")
            print(f"  Memory: {gpu_info['total_memory_gb']:.2f} GB")
            print(f"  Capability: {gpu_info['capability']}")
    else:
        print("No CUDA devices available - using CPU")
    
    print("\nOptimizations:")
    if torch.cuda.is_available():
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test device detection
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print_device_summary()
    
    device = get_device()
    print(f"Selected device: {device}")
    
    batch_size, num_workers = auto_batch_and_workers()
    print(f"Recommended: batch_size={batch_size}, num_workers={num_workers}")
    
    enable_cuda_optimizations()
    
    config = get_dataloader_config()
    print(f"DataLoader config: {config}")
