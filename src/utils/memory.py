"""
Memory management utilities for OOM safety.
"""

import torch
import gc
import psutil
from typing import Tuple, Optional
import numpy as np


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get GPU memory information.
    
    Returns:
        (allocated_gb, reserved_gb, free_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    
    return allocated, reserved, free


def get_cpu_memory_info() -> Tuple[float, float]:
    """
    Get CPU memory information.
    
    Returns:
        (used_gb, available_gb)
    """
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    available_gb = mem.available / 1024**3
    return used_gb, available_gb


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def estimate_safe_batch_size(model: torch.nn.Module, input_shape: Tuple[int, ...],
                               max_memory_fraction: float = 0.8) -> int:
    """
    Estimate safe batch size based on available GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of single input (C, H, W)
        max_memory_fraction: Maximum fraction of GPU memory to use
        
    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 4  # Conservative default for CPU
    
    device = next(model.parameters()).device
    
    # Clear cache first
    clear_cuda_cache()
    
    # Get available memory
    _, _, free_gb = get_gpu_memory_info()
    target_memory_gb = free_gb * max_memory_fraction
    
    # Estimate memory per sample with a forward+backward pass
    try:
        model.train()
        dummy_input = torch.randn(1, *input_shape, device=device)
        
        # Forward pass
        output = model(dummy_input)
        
        # Estimate backward pass (roughly 2x forward)
        if isinstance(output, torch.Tensor):
            loss = output.sum()
            loss.backward()
        
        # Memory used
        allocated, _, _ = get_gpu_memory_info()
        memory_per_sample = allocated * 1.5  # Add safety margin
        
        # Clean up
        del dummy_input, output
        if 'loss' in locals():
            del loss
        model.zero_grad()
        clear_cuda_cache()
        
        # Calculate batch size
        batch_size = max(1, int(target_memory_gb / memory_per_sample))
        
        # Cap at reasonable maximum
        batch_size = min(batch_size, 64)
        
        return batch_size
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            clear_cuda_cache()
            return 1
        raise


def reduce_batch_size(current_batch_size: int, factor: float = 0.5) -> int:
    """Reduce batch size by factor (minimum 1)."""
    new_size = max(1, int(current_batch_size * factor))
    return new_size


def tile_image(image: torch.Tensor, tile_size: int = 256, 
               overlap: int = 32) -> Tuple[list, list]:
    """
    Split image into overlapping tiles for inference.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        tiles: List of (C, tile_size, tile_size) tensors
        positions: List of (row, col) positions
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    B, C, H, W = image.shape
    stride = tile_size - overlap
    
    tiles = []
    positions = []
    
    for b in range(B):
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                # Extract tile with padding if needed
                i_end = min(i + tile_size, H)
                j_end = min(j + tile_size, W)
                
                tile = image[b:b+1, :, i:i_end, j:j_end]
                
                # Pad if needed
                if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                    pad_h = tile_size - tile.shape[2]
                    pad_w = tile_size - tile.shape[3]
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                tiles.append(tile)
                positions.append((b, i, j, i_end, j_end))
    
    return tiles, positions


def stitch_tiles(tiles: list, positions: list, output_shape: Tuple[int, ...],
                 overlap: int = 32) -> torch.Tensor:
    """
    Stitch tiles back into full image with Gaussian blending.
    
    Args:
        tiles: List of (B, C, H, W) tensors
        positions: List of (batch, row, col, row_end, col_end) tuples
        output_shape: (B, C, H, W) shape of output
        overlap: Overlap between tiles
        
    Returns:
        output: (B, C, H, W) stitched image
    """
    B, C, H, W = output_shape
    device = tiles[0].device
    
    output = torch.zeros(output_shape, device=device)
    weights = torch.zeros(output_shape, device=device)
    
    # Create Gaussian weight map for blending
    tile_size = tiles[0].shape[2]
    weight_map = create_gaussian_weight_map(tile_size, overlap).to(device)
    
    for tile, (b, i, j, i_end, j_end) in zip(tiles, positions):
        h = i_end - i
        w = j_end - j
        
        # Crop tile if it was padded
        tile_cropped = tile[:, :, :h, :w]
        weight_cropped = weight_map[:h, :w]
        
        # Accumulate
        output[b, :, i:i_end, j:j_end] += tile_cropped.squeeze(0) * weight_cropped
        weights[b, :, i:i_end, j:j_end] += weight_cropped
    
    # Normalize
    output = output / (weights + 1e-8)
    
    return output


def create_gaussian_weight_map(size: int, overlap: int) -> torch.Tensor:
    """
    Create Gaussian weight map for tile blending.
    
    Args:
        size: Tile size
        overlap: Overlap size
        
    Returns:
        weight_map: (size, size) weight map
    """
    # Create 1D Gaussian
    center = size / 2
    sigma = overlap / 3  # 3-sigma covers the overlap
    
    x = torch.arange(size, dtype=torch.float32)
    y = torch.arange(size, dtype=torch.float32)
    
    x_weights = torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    y_weights = torch.exp(-((y - center) ** 2) / (2 * sigma ** 2))
    
    # Make sure edges have lower weight for blending
    x_weights = torch.clamp(x_weights, 0.1, 1.0)
    y_weights = torch.clamp(y_weights, 0.1, 1.0)
    
    # Create 2D weight map
    weight_map = x_weights.unsqueeze(1) * y_weights.unsqueeze(0)
    
    return weight_map


def check_oom_risk(model: torch.nn.Module, batch_size: int, 
                   input_shape: Tuple[int, ...]) -> bool:
    """
    Check if current configuration risks OOM.
    
    Returns:
        True if OOM risk is high
    """
    if not torch.cuda.is_available():
        return False
    
    # Get free memory
    _, _, free_gb = get_gpu_memory_info()
    
    # Estimate memory needed (rough heuristic)
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    
    # Input/output memory (forward + backward ≈ 3x activation memory)
    C, H, W = input_shape
    activation_memory = batch_size * C * H * W * 4 / 1024**3  # 4 bytes per float32
    activation_memory *= 3  # Forward + backward
    
    total_needed = param_memory + activation_memory
    
    # Add safety margin
    total_needed *= 1.5
    
    return total_needed > free_gb * 0.9


def get_optimal_num_workers() -> int:
    """Get optimal number of workers for data loading."""
    cpu_count = psutil.cpu_count(logical=False) or 4
    return min(cpu_count, 8)  # Cap at 8 to avoid overhead


class MemoryMonitor:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.start_mem = 0
        
    def __enter__(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            clear_cuda_cache()
            self.start_mem = torch.cuda.memory_allocated() / 1024**3
        return self
    
    def __exit__(self, *args):
        if self.device == 'cuda' and torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated() / 1024**3
            delta = end_mem - self.start_mem
            print(f"Memory delta: {delta:.2f} GB (Start: {self.start_mem:.2f} GB, End: {end_mem:.2f} GB)")
