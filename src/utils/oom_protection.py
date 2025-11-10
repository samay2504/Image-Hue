"""
Out-of-memory (OOM) protection utilities for GPU training.

Provides automatic batch size reduction and gradient accumulation
to handle CUDA OOM errors gracefully on 12GB VRAM (RTX 5070 Ti).
"""

import torch
import logging
from typing import Callable, Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class OOMHandler:
    """
    Handles CUDA out-of-memory errors by automatically reducing batch size.
    
    Usage:
        handler = OOMHandler(initial_batch_size=32, min_batch_size=4)
        
        while handler.should_continue():
            try:
                loss = handler.run_training_step(train_fn, batch)
                handler.on_success()
            except RuntimeError as e:
                if handler.is_oom_error(e):
                    handler.on_oom()
                else:
                    raise
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 4,
        reduction_factor: float = 0.5,
        max_retries: int = 3,
        enable_grad_accumulation: bool = True
    ):
        """
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            reduction_factor: Factor to reduce batch size by (0.5 = halve)
            max_retries: Maximum number of OOM recovery attempts
            enable_grad_accumulation: Use gradient accumulation to maintain effective batch size
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.max_retries = max_retries
        self.enable_grad_accumulation = enable_grad_accumulation
        
        self.retry_count = 0
        self.grad_accum_steps = 1
        self._success = False
    
    @staticmethod
    def is_oom_error(exception: Exception) -> bool:
        """Check if exception is a CUDA OOM error."""
        if not isinstance(exception, RuntimeError):
            return False
        
        msg = str(exception).lower()
        return "out of memory" in msg or "cuda" in msg and "memory" in msg
    
    def on_oom(self):
        """Handle OOM error by reducing batch size."""
        if self.retry_count >= self.max_retries:
            logger.error(f"OOM: Exceeded max retries ({self.max_retries})")
            raise RuntimeError(
                f"CUDA OOM: Unable to recover after {self.max_retries} attempts. "
                f"Current batch_size={self.current_batch_size}, min={self.min_batch_size}"
            )
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        # Reduce batch size
        new_batch_size = int(self.current_batch_size * self.reduction_factor)
        new_batch_size = max(new_batch_size, self.min_batch_size)
        
        if new_batch_size >= self.current_batch_size:
            logger.error(f"OOM: Cannot reduce batch size further (current: {self.current_batch_size}, min: {self.min_batch_size})")
            raise RuntimeError(
                f"CUDA OOM: Batch size already at minimum ({self.min_batch_size}). "
                "Try reducing model size or image resolution."
            )
        
        # Update gradient accumulation steps if enabled
        if self.enable_grad_accumulation:
            # Maintain effective batch size = current_batch_size * grad_accum_steps
            old_effective = self.current_batch_size * self.grad_accum_steps
            self.grad_accum_steps = max(1, old_effective // new_batch_size)
        
        logger.warning(
            f"OOM Recovery (attempt {self.retry_count + 1}/{self.max_retries}): "
            f"Reducing batch_size {self.current_batch_size} → {new_batch_size}"
        )
        if self.enable_grad_accumulation:
            logger.info(f"  Gradient accumulation steps: {self.grad_accum_steps}")
            logger.info(f"  Effective batch size: {new_batch_size * self.grad_accum_steps}")
        
        self.current_batch_size = new_batch_size
        self.retry_count += 1
    
    def on_success(self):
        """Mark operation as successful."""
        self._success = True
        if self.retry_count > 0:
            logger.info(
                f"OOM Recovery successful! Using batch_size={self.current_batch_size}, "
                f"grad_accum_steps={self.grad_accum_steps}"
            )
    
    def should_continue(self) -> bool:
        """Check if we should retry after OOM."""
        return self.retry_count < self.max_retries
    
    def get_current_config(self) -> Dict[str, int]:
        """Get current batch size and gradient accumulation configuration."""
        return {
            'batch_size': self.current_batch_size,
            'grad_accum_steps': self.grad_accum_steps,
            'effective_batch_size': self.current_batch_size * self.grad_accum_steps
        }
    
    def reset(self):
        """Reset handler for next batch/epoch."""
        self.retry_count = 0
        self._success = False


def safe_forward_backward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    current_step: int = 0
) -> Tuple[torch.Tensor, bool]:
    """
    Safely run forward and backward pass with OOM protection.
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        targets: Target tensor
        loss_fn: Loss function
        optimizer: Optimizer
        scaler: GradScaler for AMP (optional)
        use_amp: Use automatic mixed precision
        grad_accum_steps: Number of gradient accumulation steps
        current_step: Current training step (for grad accumulation)
    
    Returns:
        Tuple of (loss, success)
    """
    try:
        # Forward pass with AMP
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        
        # Backward pass
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only at accumulation boundary
        if (current_step + 1) % grad_accum_steps == 0:
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        return loss * grad_accum_steps, True  # Return unscaled loss
    
    except RuntimeError as e:
        if OOMHandler.is_oom_error(e):
            logger.warning(f"OOM in forward/backward pass: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, False
        else:
            raise


def estimate_max_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    start_batch_size: int = 64,
    safety_factor: float = 0.8
) -> int:
    """
    Estimate maximum batch size by binary search with OOM testing.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        device: Device to test on
        start_batch_size: Starting batch size for search
        safety_factor: Multiply result by this factor for safety margin
    
    Returns:
        Maximum safe batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning conservative batch size")
        return 8
    
    model.eval()
    model.to(device)
    
    # Binary search
    low, high = 1, start_batch_size
    max_working = 1
    
    logger.info(f"Estimating max batch size (starting from {start_batch_size})...")
    
    while low <= high:
        mid = (low + high) // 2
        
        try:
            # Test forward + backward pass
            torch.cuda.empty_cache()
            dummy_input = torch.randn(mid, *input_shape, device=device)
            
            with torch.cuda.amp.autocast():
                output = model(dummy_input)
                loss = output.mean()
            
            loss.backward()
            
            # Success - try larger
            max_working = mid
            logger.info(f"  batch_size={mid}: ✓ (working)")
            low = mid + 1
            
        except RuntimeError as e:
            if OOMHandler.is_oom_error(e):
                # OOM - try smaller
                logger.info(f"  batch_size={mid}: ✗ (OOM)")
                high = mid - 1
            else:
                raise
        
        finally:
            torch.cuda.empty_cache()
    
    # Apply safety factor
    safe_batch_size = int(max_working * safety_factor)
    logger.info(f"Max batch size: {max_working}, safe batch size (×{safety_factor}): {safe_batch_size}")
    
    model.train()
    return safe_batch_size


if __name__ == "__main__":
    # Test OOM handler
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 60)
    print("OOM Handler Test")
    print("=" * 60)
    
    handler = OOMHandler(initial_batch_size=64, min_batch_size=4)
    
    print(f"\nInitial config: {handler.get_current_config()}")
    
    # Simulate 2 OOM errors
    for i in range(2):
        print(f"\n--- Simulating OOM #{i+1} ---")
        handler.on_oom()
        print(f"Current config: {handler.get_current_config()}")
    
    handler.on_success()
    print("\n✓ Handler test completed successfully!")
