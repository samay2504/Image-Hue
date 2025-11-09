"""
Safe-mode utilities and OOM handling for modern colorization.

Provides automatic fallback strategies, memory profiling, and cleanup handlers.
"""

import logging
import gc
import torch
import psutil
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "total": total,
        "free": total - allocated,
    }


def get_ram_memory_info() -> Dict[str, float]:
    """Get current RAM usage in GB."""
    mem = psutil.virtual_memory()
    return {
        "used": mem.used / 1e9,
        "available": mem.available / 1e9,
        "total": mem.total / 1e9,
        "percent": mem.percent,
    }


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def safe_model_forward(
    model: torch.nn.Module,
    *args,
    fallback_fn: Optional[Callable] = None,
    cleanup_on_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely run model forward with OOM handling.
    
    Args:
        model: PyTorch model
        *args: Positional arguments for forward
        fallback_fn: Function to call on OOM (should take same args)
        cleanup_on_error: Run memory cleanup on error
        **kwargs: Keyword arguments for forward
    
    Returns:
        Model output or fallback output
    
    Raises:
        RuntimeError: If fallback also fails
    """
    try:
        return model(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"OOM error during forward pass: {e}")
            logger.info("Running memory cleanup...")
            
            if cleanup_on_error:
                cleanup_memory()
            
            if fallback_fn is not None:
                logger.info("Attempting fallback strategy")
                try:
                    return fallback_fn(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            else:
                logger.error("No fallback available, re-raising OOM")
                raise
        else:
            raise


def with_oom_protection(fallback_fn: Optional[Callable] = None):
    """
    Decorator for OOM-protected functions.
    
    Usage:
        @with_oom_protection(fallback_fn=my_fallback)
        def my_function(x):
            return model(x)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM in {func.__name__}: {e}")
                    cleanup_memory()
                    
                    if fallback_fn:
                        logger.info(f"Using fallback for {func.__name__}")
                        return fallback_fn(*args, **kwargs)
                    raise
                else:
                    raise
        return wrapper
    return decorator


class SafeModeContext:
    """
    Context manager for safe-mode execution with memory monitoring.
    
    Usage:
        with SafeModeContext(memory_threshold=0.9) as safe:
            output = model(input)
            safe.check_memory()  # Raises if threshold exceeded
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.9,
        log_memory: bool = True,
        cleanup_on_exit: bool = True,
    ):
        """
        Args:
            memory_threshold: Fraction of GPU memory before warning (0-1)
            log_memory: Log memory usage on enter/exit
            cleanup_on_exit: Run cleanup on context exit
        """
        self.memory_threshold = memory_threshold
        self.log_memory = log_memory
        self.cleanup_on_exit = cleanup_on_exit
        self.initial_memory = None
    
    def __enter__(self):
        if self.log_memory:
            self.initial_memory = get_gpu_memory_info()
            logger.info(
                f"GPU memory at entry: "
                f"{self.initial_memory['allocated']:.2f}GB / "
                f"{self.initial_memory['total']:.2f}GB "
                f"({self.initial_memory['allocated']/self.initial_memory['total']*100:.1f}%)"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_memory:
            final_memory = get_gpu_memory_info()
            delta = final_memory['allocated'] - self.initial_memory['allocated']
            logger.info(
                f"GPU memory at exit: "
                f"{final_memory['allocated']:.2f}GB / "
                f"{final_memory['total']:.2f}GB "
                f"(Δ {delta:+.2f}GB)"
            )
        
        if self.cleanup_on_exit:
            cleanup_memory()
        
        return False  # Don't suppress exceptions
    
    def check_memory(self):
        """Check if memory usage exceeds threshold."""
        mem = get_gpu_memory_info()
        usage_fraction = mem['allocated'] / mem['total']
        
        if usage_fraction > self.memory_threshold:
            logger.warning(
                f"Memory usage {usage_fraction:.1%} exceeds threshold "
                f"{self.memory_threshold:.1%}"
            )
            raise RuntimeError(
                f"Memory threshold exceeded: {usage_fraction:.1%} > "
                f"{self.memory_threshold:.1%}"
            )


class ModelSizeScaler:
    """
    Automatically scale model size based on available memory.
    
    Provides fallback strategy from large models to small models
    based on GPU memory availability.
    """
    
    def __init__(self, available_memory_gb: Optional[float] = None):
        """
        Args:
            available_memory_gb: Available GPU memory in GB
                                (auto-detected if None)
        """
        if available_memory_gb is None:
            mem = get_gpu_memory_info()
            available_memory_gb = mem['total']
        
        self.available_memory = available_memory_gb
        logger.info(f"Available GPU memory: {self.available_memory:.2f}GB")
    
    def get_recommended_encoder_size(self) -> str:
        """Get recommended encoder size based on available memory."""
        if self.available_memory >= 10.0:
            return "base"  # ViT-Base or Swin-Base
        elif self.available_memory >= 6.0:
            return "tiny"  # ViT-Tiny
        else:
            logger.warning(
                f"Low memory ({self.available_memory:.1f}GB), "
                "consider using CPU or paper VGG model"
            )
            return "tiny"
    
    def get_recommended_batch_size(
        self,
        encoder_size: str,
        image_size: int = 256,
    ) -> int:
        """Get recommended batch size based on encoder and memory."""
        # Rough estimates based on 256x256 images
        memory_per_sample = {
            "tiny": 0.35,   # ~350MB per sample
            "base": 0.85,   # ~850MB per sample
            "swin": 0.90,   # ~900MB per sample
        }
        
        mem_per_sample = memory_per_sample.get(encoder_size, 0.85)
        
        # Scale for image size (quadratic)
        scale_factor = (image_size / 256) ** 2
        mem_per_sample *= scale_factor
        
        # Use 80% of available memory
        usable_memory = self.available_memory * 0.8
        batch_size = int(usable_memory / mem_per_sample)
        
        # Clamp to reasonable range
        batch_size = max(1, min(batch_size, 32))
        
        logger.info(
            f"Recommended batch size for {encoder_size} @ {image_size}x{image_size}: "
            f"{batch_size}"
        )
        
        return batch_size


def log_system_info():
    """Log comprehensive system and memory information."""
    logger.info("="*60)
    logger.info("System Information")
    logger.info("="*60)
    
    # CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        gpu_mem = get_gpu_memory_info()
        logger.info(f"GPU memory total: {gpu_mem['total']:.2f}GB")
        logger.info(f"GPU memory allocated: {gpu_mem['allocated']:.2f}GB")
        logger.info(f"GPU memory free: {gpu_mem['free']:.2f}GB")
    else:
        logger.info("CUDA available: No")
    
    # RAM info
    ram = get_ram_memory_info()
    logger.info(f"RAM total: {ram['total']:.2f}GB")
    logger.info(f"RAM used: {ram['used']:.2f}GB ({ram['percent']:.1f}%)")
    logger.info(f"RAM available: {ram['available']:.2f}GB")
    
    # PyTorch info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Num threads: {torch.get_num_threads()}")
    
    logger.info("="*60)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Log system info
    log_system_info()
    
    # Test memory monitoring
    print("\nTesting memory monitoring:")
    with SafeModeContext(memory_threshold=0.9) as safe:
        print("  Inside safe context")
        mem = get_gpu_memory_info()
        print(f"  GPU: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB")
    
    # Test model size scaling
    print("\nTesting model size scaler:")
    scaler = ModelSizeScaler()
    size = scaler.get_recommended_encoder_size()
    print(f"  Recommended encoder: {size}")
    batch_size = scaler.get_recommended_batch_size(size)
    print(f"  Recommended batch size: {batch_size}")
