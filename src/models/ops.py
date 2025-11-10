"""
Quantization and color space operations for image colorization.

Implements the key operations from "Colorful Image Colorization" (Zhang et al., ECCV 2016):
- CIE Lab color space conversion
- ab space quantization into 313 bins (grid size 10)
- Soft-encoding with Gaussian kernel (σ=5)
- Class rebalancing weights computation
- Annealed-mean decoding (Equation 5 from paper)
"""

import numpy as np
from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial

# Constants from paper
GRID_SIZE = 10  # Quantization grid size
SIGMA_SOFT = 5.0  # Gaussian kernel σ for soft-encoding
SIGMA_REBALANCE = 5.0  # Gaussian smoothing σ for class rebalancing
LAMBDA_REBALANCE = 0.5  # Mix parameter for uniform distribution
K_NEIGHBORS = 5  # Number of nearest neighbors for soft-encoding
DEFAULT_TEMPERATURE = 0.38  # Default annealed-mean temperature from paper


def get_ab_quantization_grid(grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Generate the ab quantization grid and filter to in-gamut colors.
    
    Returns:
        ab_grid: (Q, 2) array of ab bin centers, where Q=313 for grid_size=10
    """
    # Generate full grid
    a_range = np.arange(-110, 110, grid_size)
    b_range = np.arange(-110, 110, grid_size)
    aa, bb = np.meshgrid(a_range, b_range)
    ab_grid = np.stack([aa.flatten(), bb.flatten()], axis=1)
    
    # Filter to in-gamut colors (those that can be represented in RGB)
    # Use multiple L values to ensure color is representable
    # in_gamut = []
    # for ab in ab_grid:
    #     # Test with L=50 (mid-lightness) - this is standard
    #     lab = np.array([50.0, ab[0], ab[1]])
    #     rgb = lab_to_rgb_single(lab)
    #     # Must be in valid RGB range [0, 1] with small tolerance
    #     if np.all(rgb >= 0.0) and np.all(rgb <= 1.0):
    #         in_gamut.append(ab)
    in_gamut = []
    for ab in ab_grid:
        lab = np.array([50.0, ab[0], ab[1]])
        if is_color_in_gamut(lab, tolerance=0.3):
            in_gamut.append(ab)
    if len(in_gamut) != 313:
        print(f"Warning: Got {len(in_gamut)} bins, forcing to 313")
        # Sort by distance from origin and take first 313
        ab_array = np.array(in_gamut)
        distances = np.sqrt(ab_array[:, 0]**2 + ab_array[:, 1]**2)
        sorted_indices = np.argsort(distances)
        in_gamut = ab_array[sorted_indices[:313]].tolist()

    ab_grid_filtered = np.array(in_gamut)
    
    # Paper reports Q=313 bins for grid_size=10
    return ab_grid_filtered.astype(np.float32)
    

# Pre-compute and cache the quantization grid
_AB_GRID_CACHE: Optional[np.ndarray] = None


def reset_ab_grid_cache():
    """Reset the cached ab grid (useful for testing)."""
    global _AB_GRID_CACHE
    _AB_GRID_CACHE = None


def get_ab_grid(grid_size: int = GRID_SIZE) -> np.ndarray:
    """Get cached ab quantization grid."""
    global _AB_GRID_CACHE
    if _AB_GRID_CACHE is None or len(_AB_GRID_CACHE) == 0:
        _AB_GRID_CACHE = get_ab_quantization_grid(grid_size)
    return _AB_GRID_CACHE


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIE Lab color space.
    
    Args:
        rgb: (H, W, 3) array in [0, 1] or (3, H, W) tensor format
        
    Returns:
        lab: (H, W, 3) array with L in [0, 100], a in [-127, 127], b in [-127, 127]
    """
    # Handle tensor format
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    
    rgb = np.clip(rgb, 0, 1)
    
    # RGB to XYZ (using D65 illuminant)
    # First apply gamma correction (inverse sRGB companding)
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    
    # XYZ conversion matrix (D65 illuminant)
    xyz = np.dot(rgb_linear, np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]).T)
    
    # XYZ to Lab (D65 white point: [95.047, 100.0, 108.883])
    xyz = xyz / np.array([95.047, 100.0, 108.883]) * 100
    
    mask = xyz > 0.008856
    xyz_f = np.where(mask, np.power(xyz, 1/3), 7.787 * xyz + 16/116)
    
    L = 116 * xyz_f[:, :, 1] - 16
    a = 500 * (xyz_f[:, :, 0] - xyz_f[:, :, 1])
    b = 200 * (xyz_f[:, :, 1] - xyz_f[:, :, 2])
    
    lab = np.stack([L, a, b], axis=2)
    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE Lab to RGB color space.
    
    Args:
        lab: (H, W, 3) array with L in [0, 100], a in [-127, 127], b in [-127, 127]
        
    Returns:
        rgb: (H, W, 3) array in [0, 1]
    """
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # Lab to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    xyz_f = np.stack([fx, fy, fz], axis=2)
    
    # Inverse transform
    mask = xyz_f > 0.206893
    xyz = np.where(mask, np.power(xyz_f, 3), (xyz_f - 16/116) / 7.787)
    xyz = xyz * np.array([95.047, 100.0, 108.883]) / 100
    
    # XYZ to RGB
    rgb_linear = np.dot(xyz, np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ]).T)
    
    # Apply sRGB companding
    # mask = rgb_linear > 0.0031308
    # rgb = np.where(mask, 1.055 * np.power(np.maximum(rgb_linear, 0), 1/2.4) - 0.055, 12.92 * rgb_linear)
    
    # rgb = np.clip(rgb, 0, 1)
    # return rgb.astype(np.float32)
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.maximum(rgb_linear, 0), 1/2.4) - 0.055, 12.92 * rgb_linear)

    # Check if in gamut BEFORE clipping (this is what the filter should see)
    # For filtering purposes, we don't clip - let the filter see the real out-of-gamut values
    # But for actual use, we do clip
    return rgb.astype(np.float32)


def lab_to_rgb_single(lab: np.ndarray) -> np.ndarray:
    """Convert single Lab color to RGB."""
    lab_img = lab.reshape(1, 1, 3)
    rgb_img = lab_to_rgb(lab_img)
    return rgb_img[0, 0]
def is_color_in_gamut(lab: np.ndarray, tolerance: float = 0.01) -> bool:
    """
    Check if a Lab color is within the sRGB gamut.
    
    Args:
        lab: (3,) array [L, a, b]
        tolerance: Allowed overshoot/undershoot
        
    Returns:
        True if color is in gamut, False otherwise
    """
    # Convert Lab to XYZ
    L, a, b = lab
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    xyz_f = np.array([fx, fy, fz])
    
    # Inverse transform
    mask = xyz_f > 0.206893
    xyz = np.where(mask, xyz_f ** 3, (xyz_f - 16/116) / 7.787)
    xyz = xyz * np.array([95.047, 100.0, 108.883]) / 100
    
    # XYZ to RGB linear
    rgb_linear = np.dot(xyz, np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ]))
    
    # Check if in gamut (before any clipping or NaN handling)
    if np.any(np.isnan(rgb_linear)) or np.any(np.isinf(rgb_linear)):
        return False
    
    # Allow small tolerance for floating point errors
    return np.all(rgb_linear >= -tolerance) and np.all(rgb_linear <= 1.0 + tolerance)

def encode_ab_to_distribution(ab: np.ndarray, ab_grid: Optional[np.ndarray] = None,
                                sigma: float = SIGMA_SOFT, k: int = K_NEIGHBORS) -> np.ndarray:
    """
    Soft-encode ground truth ab values to distribution over Q bins.
    
    Uses Gaussian kernel with σ=5 to weight k=5 nearest neighbors.
    
    Args:
        ab: (H, W, 2) array of ground truth ab values
        ab_grid: (Q, 2) array of ab bin centers (default: use cached grid)
        sigma: Gaussian kernel standard deviation
        k: Number of nearest neighbors
        
    Returns:
        dist: (H, W, Q) distribution (sums to 1 along Q dimension)
    """
    if ab_grid is None:
        ab_grid = get_ab_grid()
    
    H, W = ab.shape[:2]
    Q = len(ab_grid)
    
    # Flatten spatial dimensions
    ab_flat = ab.reshape(-1, 2)  # (H*W, 2)
    
    # Compute distances to all bins
    # ab_flat: (H*W, 2), ab_grid: (Q, 2)
    # distances: (H*W, Q)
    distances = np.sqrt(np.sum((ab_flat[:, np.newaxis, :] - ab_grid[np.newaxis, :, :]) ** 2, axis=2))
    
    # Find k nearest neighbors
    nearest_indices = np.argsort(distances, axis=1)[:, :k]  # (H*W, k)
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)  # (H*W, k)
    
    # Compute Gaussian weights
    weights = np.exp(-nearest_distances ** 2 / (2 * sigma ** 2))  # (H*W, k)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)  # Normalize
    
    # Create distribution
    dist = np.zeros((H * W, Q), dtype=np.float32)
    np.put_along_axis(dist, nearest_indices, weights, axis=1)
    
    dist = dist.reshape(H, W, Q)
    return dist


def decode_distribution_to_ab(dist: torch.Tensor, ab_grid: Optional[np.ndarray] = None,
                                temperature: float = DEFAULT_TEMPERATURE) -> torch.Tensor:
    """
    Decode distribution over Q bins to ab values using annealed-mean (Equation 5 from paper).
    
    Args:
        dist: (B, Q, H, W) or (Q, H, W) tensor of logits or probabilities
        ab_grid: (Q, 2) array of ab bin centers
        temperature: Temperature for annealing (default: 0.38)
        
    Returns:
        ab: (B, 2, H, W) or (2, H, W) tensor of predicted ab values
    """
    if ab_grid is None:
        ab_grid = get_ab_grid()
    
    # Convert to tensor if needed
    if isinstance(ab_grid, np.ndarray):
        ab_grid_t = torch.from_numpy(ab_grid).to(dist.device)
    else:
        ab_grid_t = ab_grid
    
    # Handle different input shapes
    if dist.ndim == 3:  # (Q, H, W)
        dist = dist.unsqueeze(0)  # (1, Q, H, W)
        squeeze = True
    else:
        squeeze = False
    
    B, Q, H, W = dist.shape
    
    # Apply temperature and softmax (annealed-mean, Equation 5)
    if temperature != 1.0:
        dist = dist / temperature
    probs = F.softmax(dist, dim=1)  # (B, Q, H, W)
    
    # Compute expected value: E[ab] = Σ p(q) * ab(q)
    # probs: (B, Q, H, W), ab_grid: (Q, 2)
    # Result: (B, 2, H, W)
    probs_flat = probs.permute(0, 2, 3, 1).reshape(B * H * W, Q)  # (B*H*W, Q)
    ab_grid_expanded = ab_grid_t.unsqueeze(0).expand(B * H * W, -1, -1)  # (B*H*W, Q, 2)
    
    # Weighted sum
    ab_pred = torch.sum(probs_flat.unsqueeze(2) * ab_grid_expanded, dim=1)  # (B*H*W, 2)
    ab_pred = ab_pred.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # (B, 2, H, W)
    
    if squeeze:
        ab_pred = ab_pred.squeeze(0)  # (2, H, W)
    
    return ab_pred


def compute_class_rebalancing_weights(empirical_distribution: np.ndarray,
                                       sigma: float = SIGMA_REBALANCE,
                                       lambda_mix: float = LAMBDA_REBALANCE) -> np.ndarray:
    """
    Compute class rebalancing weights as described in the paper.
    
    Process:
    1. Smooth empirical distribution with Gaussian (σ=5)
    2. Mix with uniform distribution: p̃ = (1-λ)*p_smooth + λ*(1/Q)
    3. Compute weights: w ∝ p̃^(-1)
    4. Normalize so E[w] = 1
    
    Args:
        empirical_distribution: (Q,) array of empirical color frequencies
        sigma: Gaussian smoothing σ
        lambda_mix: Mixing parameter λ
        
    Returns:
        weights: (Q,) array of class weights
    """
    Q = len(empirical_distribution)
    
    # Normalize empirical distribution
    p_empirical = empirical_distribution / (empirical_distribution.sum() + 1e-8)
    
    # Smooth with Gaussian
    p_smooth = gaussian_filter(p_empirical, sigma=sigma, mode='constant')
    p_smooth = p_smooth / (p_smooth.sum() + 1e-8)
    
    # Mix with uniform distribution
    p_uniform = np.ones(Q) / Q
    p_tilde = (1 - lambda_mix) * p_smooth + lambda_mix * p_uniform
    
    # Compute weights: w ∝ p̃^(-1)
    weights = 1.0 / (p_tilde + 1e-8)
    
    # Normalize so E[w] = 1
    weights = weights / weights.mean()
    
    return weights.astype(np.float32)

def _process_single_image(img_path: str, ab_grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Helper function to process a single image and return its histogram.
    This is separated out for parallel processing.
    """
    try:
        from PIL import Image
        # Load and convert to Lab
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.0
        lab = rgb_to_lab(img)
        ab = lab[:, :, 1:]  # (H, W, 2)
        
        # Quantize to nearest bin
        ab_flat = ab.reshape(-1, 2)
        distances = np.sqrt(np.sum((ab_flat[:, np.newaxis, :] - ab_grid[np.newaxis, :, :]) ** 2, axis=2))
        nearest_bins = np.argmin(distances, axis=1)
        
        # Compute histogram
        Q = len(ab_grid)
        hist, _ = np.histogram(nearest_bins, bins=np.arange(Q + 1))
        return hist
        
    except Exception as e:
        print(f"Warning: Failed to process {img_path}: {e}")
        return None



#GPU TRIAL  
def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to Lab color space (GPU version).
    
    Args:
        rgb: (H, W, 3) or (B, H, W, 3) tensor in [0, 1]
        
    Returns:
        lab: (H, W, 3) or (B, H, W, 3) tensor
    """
    # XYZ conversion matrix
    rgb_to_xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)
    
    # Linearize RGB
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, 
                             torch.pow((rgb + 0.055) / 1.055, 2.4),
                             rgb / 12.92)
    
    # Convert to XYZ
    xyz = torch.matmul(rgb_linear, rgb_to_xyz.T)
    
    # Normalize by D65 white point
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], 
                             dtype=xyz.dtype, device=xyz.device)
    
    # XYZ to Lab
    epsilon = 0.008856
    kappa = 903.3
    
    mask = xyz > epsilon
    f = torch.where(mask, torch.pow(xyz, 1/3), (kappa * xyz + 16) / 116)
    
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    
    return torch.stack([L, a, b], dim=-1)

#gpu trial
def compute_empirical_distribution_from_images_gpu(
    image_paths: List[str],
    ab_grid: Optional[np.ndarray] = None,
    max_images: int = 10000,
    batch_size: int = 32,
    device: str = 'cuda'
) -> np.ndarray:
    """
    GPU-accelerated color statistics computation.
    
    Args:
        image_paths: List of paths to color images
        ab_grid: (Q, 2) array of ab bin centers
        max_images: Maximum number of images to process
        batch_size: Number of images to process in parallel on GPU
        device: 'cuda' or 'cpu'
        
    Returns:
        empirical_dist: (Q,) array of color frequencies
    """
    from PIL import Image
    import torch
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    if ab_grid is None:
        ab_grid = get_ab_grid()
    
    # Move ab_grid to GPU
    ab_grid_torch = torch.from_numpy(ab_grid).float().to(device)  # (Q, 2)
    Q = len(ab_grid)
    
    empirical_dist = np.zeros(Q, dtype=np.float64)
    image_paths = image_paths[:max_images]
    
    print(f"Processing on {device.upper()} with batch_size={batch_size}")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(image_paths), batch_size),
                           desc="Processing batches", unit="batch"):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_images = []
        
        # Load batch of images (CPU)
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize to reduce memory
                img.thumbnail((256, 256), Image.LANCZOS)
                img_array = np.array(img) / 255.0
                batch_images.append(img_array)
            except:
                continue
        
        if not batch_images:
            continue
        
        # Stack into batch tensor and move to GPU
        # Pad to same size if needed
        max_h = max(img.shape[0] for img in batch_images)
        max_w = max(img.shape[1] for img in batch_images)
        
        batch_tensor = []
        for img in batch_images:
            h, w = img.shape[:2]
            if h != max_h or w != max_w:
                # Pad
                padded = np.zeros((max_h, max_w, 3))
                padded[:h, :w] = img
                batch_tensor.append(padded)
            else:
                batch_tensor.append(img)
        
        batch_tensor = torch.from_numpy(np.array(batch_tensor)).float().to(device)
        # batch_tensor: (B, H, W, 3)
        
        # Convert to Lab on GPU
        with torch.no_grad():
            lab = rgb_to_lab_torch(batch_tensor)  # (B, H, W, 3)
            ab = lab[..., 1:]  # (B, H, W, 2)
            
            # Flatten spatial dimensions
            B, H, W, _ = ab.shape
            ab_flat = ab.reshape(B * H * W, 2)  # (N, 2)
            
            # Compute distances to all bins - THIS IS THE GPU WIN
            # ab_flat: (N, 2), ab_grid_torch: (Q, 2)
            # distances: (N, Q)
            distances = torch.cdist(ab_flat, ab_grid_torch)  # Efficient pairwise distances
            nearest_bins = torch.argmin(distances, dim=1)  # (N,)
            
            # Compute histogram on GPU
            hist = torch.histc(nearest_bins.float(), bins=Q, min=0, max=Q-1)
            
            # Move back to CPU
            empirical_dist += hist.cpu().numpy()
    
    return empirical_dist.astype(np.float32)




#tested to beginningt, cpu usage
def compute_empirical_distribution_from_images(image_paths: List[str],
                                                 ab_grid: Optional[np.ndarray] = None,
                                                 max_images: int = 10000) -> np.ndarray:
    """
    Compute empirical color distribution from a dataset of images.
    Memory-efficient sequential processing.
    
    Args:
        image_paths: List of paths to color images
        ab_grid: (Q, 2) array of ab bin centers
        max_images: Maximum number of images to process
        
    Returns:
        empirical_dist: (Q,) array of color frequencies
    """
    from PIL import Image
    
    if ab_grid is None:
        ab_grid = get_ab_grid()
    
    Q = len(ab_grid)
    empirical_dist = np.zeros(Q, dtype=np.float64)
    image_paths = image_paths[:max_images]
    
    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        try:
            # Load and convert to Lab
            img = Image.open(img_path).convert('RGB')
            
            # MEMORY FIX: Resize large images to max 512x512
            if max(img.size) > 512:
                img.thumbnail((512, 512), Image.LANCZOS)
            
            img = np.array(img) / 255.0
            lab = rgb_to_lab(img)
            ab = lab[:, :, 1:]  # (H, W, 2)
            
            # Quantize to nearest bin - MEMORY EFFICIENT VERSION
            ab_flat = ab.reshape(-1, 2)  # (N, 2)
            
            # Process in chunks to avoid memory explosion
            chunk_size = 10000
            nearest_bins = np.zeros(len(ab_flat), dtype=np.int32)
            
            for i in range(0, len(ab_flat), chunk_size):
                chunk = ab_flat[i:i+chunk_size]
                # Compute distances for this chunk only
                distances = np.sqrt(np.sum((chunk[:, np.newaxis, :] - ab_grid[np.newaxis, :, :]) ** 2, axis=2))
                nearest_bins[i:i+chunk_size] = np.argmin(distances, axis=1)
            
            # Accumulate histogram
            hist, _ = np.histogram(nearest_bins, bins=np.arange(Q + 1))
            empirical_dist += hist
            
        except Exception as e:
            # Silently skip failed images to keep progress bar clean
            continue
    
    return empirical_dist.astype(np.float32)



#this function worked but then stopped at 15 percent
# def compute_empirical_distribution_from_images(image_paths: List[str],
#                                                  ab_grid: Optional[np.ndarray] = None,
#                                                  max_images: int = 10000,
#                                                  num_workers: Optional[int] = None) -> np.ndarray:
#     """
#     Compute empirical color distribution from a dataset of images.
    
#     Args:
#         image_paths: List of paths to color images
#         ab_grid: (Q, 2) array of ab bin centers
#         max_images: Maximum number of images to process
#         num_workers: Number of parallel workers (None = auto-detect CPU count)
        
#     Returns:
#         empirical_dist: (Q,) array of color frequencies
#     """
#     import os
    
#     if ab_grid is None:
#         ab_grid = get_ab_grid()
    
#     Q = len(ab_grid)
#     empirical_dist = np.zeros(Q, dtype=np.float64)
    
#     image_paths = image_paths[:max_images]
    
#     # Auto-detect CPU count if not specified
#     if num_workers is None:
#         num_workers = min(os.cpu_count() or 1, len(image_paths))
    
#     print(f"Processing {len(image_paths)} images using {num_workers} workers...")
    
#     # Create partial function with ab_grid pre-filled
#     process_func = partial(_process_single_image, ab_grid=ab_grid)
    
#     # Use ProcessPoolExecutor for true parallelism (GIL-free)
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         # Submit all jobs
#         futures = {executor.submit(process_func, img_path): img_path 
#                    for img_path in image_paths}
        
#         # Process results as they complete with progress bar
#         for future in tqdm(as_completed(futures), total=len(image_paths), 
#                           desc="Processing images", unit="img"):
#             hist = future.result()
#             if hist is not None:
#                 empirical_dist += hist
    
#     return empirical_dist.astype(np.float32)



#original function
# def compute_empirical_distribution_from_images(image_paths: List[str],
#                                                  ab_grid: Optional[np.ndarray] = None,
#                                                  max_images: int = 10000) -> np.ndarray:
#     """
#     Compute empirical color distribution from a dataset of images.
    
#     Args:
#         image_paths: List of paths to color images
#         ab_grid: (Q, 2) array of ab bin centers
#         max_images: Maximum number of images to process
        
#     Returns:
#         empirical_dist: (Q,) array of color frequencies
#     """
#     from PIL import Image
    
#     if ab_grid is None:
#         ab_grid = get_ab_grid()
    
#     Q = len(ab_grid)
#     empirical_dist = np.zeros(Q, dtype=np.float64)
    
#     image_paths = image_paths[:max_images]
    
#     for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
#         try:
#             # Load and convert to Lab
#             img = Image.open(img_path).convert('RGB')
#             img = np.array(img) / 255.0
#             lab = rgb_to_lab(img)
#             ab = lab[:, :, 1:]  # (H, W, 2)
            
#             # Quantize to nearest bin
#             ab_flat = ab.reshape(-1, 2)
#             distances = np.sqrt(np.sum((ab_flat[:, np.newaxis, :] - ab_grid[np.newaxis, :, :]) ** 2, axis=2))
#             nearest_bins = np.argmin(distances, axis=1)
            
#             # Accumulate histogram
#             hist, _ = np.histogram(nearest_bins, bins=np.arange(Q + 1))
#             empirical_dist += hist
            
#         except Exception as e:
#             print(f"Warning: Failed to process {img_path}: {e}")
#             continue
    
#     return empirical_dist.astype(np.float32)


def ab_to_bin_indices(ab: np.ndarray, ab_grid: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert ab values to nearest bin indices.
    
    Args:
        ab: (H, W, 2) array of ab values
        ab_grid: (Q, 2) array of ab bin centers
        
    Returns:
        indices: (H, W) array of bin indices
    """
    if ab_grid is None:
        ab_grid = get_ab_grid()
    
    ab_flat = ab.reshape(-1, 2)
    distances = np.sqrt(np.sum((ab_flat[:, np.newaxis, :] - ab_grid[np.newaxis, :, :]) ** 2, axis=2))
    indices = np.argmin(distances, axis=1)
    
    return indices.reshape(ab.shape[:2])
