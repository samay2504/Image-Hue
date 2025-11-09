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
    in_gamut = []
    for ab in ab_grid:
        # Test with L=50 (mid-lightness) - this is standard
        lab = np.array([50.0, ab[0], ab[1]])
        rgb = lab_to_rgb_single(lab)
        # Must be in valid RGB range [0, 1] with small tolerance
        if np.all(rgb >= 0.0) and np.all(rgb <= 1.0):
            in_gamut.append(ab)
    
    ab_grid_filtered = np.array(in_gamut)
    
    # Paper reports Q=313 bins for grid_size=10
    return ab_grid_filtered.astype(np.float32)
    
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
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(rgb_linear, 1/2.4) - 0.055, 12.92 * rgb_linear)
    
    rgb = np.clip(rgb, 0, 1)
    return rgb.astype(np.float32)


def lab_to_rgb_single(lab: np.ndarray) -> np.ndarray:
    """Convert single Lab color to RGB."""
    lab_img = lab.reshape(1, 1, 3)
    rgb_img = lab_to_rgb(lab_img)
    return rgb_img[0, 0]


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


def compute_empirical_distribution_from_images(image_paths: List[str],
                                                 ab_grid: Optional[np.ndarray] = None,
                                                 max_images: int = 10000) -> np.ndarray:
    """
    Compute empirical color distribution from a dataset of images.
    
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
    
    for img_path in image_paths:
        try:
            # Load and convert to Lab
            img = Image.open(img_path).convert('RGB')
            img = np.array(img) / 255.0
            lab = rgb_to_lab(img)
            ab = lab[:, :, 1:]  # (H, W, 2)
            
            # Quantize to nearest bin
            ab_flat = ab.reshape(-1, 2)
            distances = np.sqrt(np.sum((ab_flat[:, np.newaxis, :] - ab_grid[np.newaxis, :, :]) ** 2, axis=2))
            nearest_bins = np.argmin(distances, axis=1)
            
            # Accumulate histogram
            hist, _ = np.histogram(nearest_bins, bins=np.arange(Q + 1))
            empirical_dist += hist
            
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            continue
    
    return empirical_dist.astype(np.float32)


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


# ============================================================================
# Token reshaping helpers for transformer encoders
# ============================================================================

def tokens_to_feature_map(
    tokens: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    remove_cls_token: bool = True,
) -> torch.Tensor:
    """
    Reshape transformer token sequence to spatial feature map.
    
    Args:
        tokens: [B, N_tokens, C] token sequence from transformer
        H: Original image height
        W: Original image width
        patch_size: Patch size used by transformer (e.g., 16)
        remove_cls_token: Whether to remove CLS token (first token)
        
    Returns:
        feature_map: [B, C, H_patch, W_patch] where
                     H_patch = H // patch_size, W_patch = W // patch_size
    """
    B, N, C = tokens.shape
    H_patch = H // patch_size
    W_patch = W // patch_size
    
    # Handle CLS token
    if remove_cls_token and N == H_patch * W_patch + 1:
        tokens = tokens[:, 1:, :]  # Remove first (CLS) token
    
    # Verify token count matches spatial dimensions
    expected_tokens = H_patch * W_patch
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"Token count mismatch: got {tokens.shape[1]}, "
            f"expected {expected_tokens} (H_patch={H_patch}, W_patch={W_patch})"
        )
    
    # Reshape: [B, N, C] -> [B, C, H_patch, W_patch]
    tokens = tokens.permute(0, 2, 1)  # [B, C, N]
    feature_map = tokens.reshape(B, C, H_patch, W_patch)
    
    return feature_map


def upsample_feature_map(
    feature: torch.Tensor,
    target_size: tuple,
    mode: str = 'bilinear',
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Upsample feature map to target spatial size.
    
    Args:
        feature: [B, C, H, W] feature map
        target_size: (H_target, W_target)
        mode: Interpolation mode ('bilinear', 'nearest', etc.)
        align_corners: Whether to align corners in bilinear interpolation
        
    Returns:
        upsampled: [B, C, H_target, W_target]
    """
    if feature.shape[2:] == target_size:
        return feature
    
    return F.interpolate(
        feature,
        size=target_size,
        mode=mode,
        align_corners=align_corners if mode == 'bilinear' else None,
    )


def multi_scale_feature_pyramid(
    features: list,
    target_sizes: list,
    mode: str = 'bilinear',
) -> list:
    """
    Resize multi-scale features to target sizes.
    
    Args:
        features: List of [B, C_i, H_i, W_i] feature tensors
        target_sizes: List of (H_target, W_target) tuples
        mode: Interpolation mode
        
    Returns:
        List of resized feature tensors
    """
    if len(features) != len(target_sizes):
        raise ValueError(
            f"Number of features ({len(features)}) must match "
            f"number of target sizes ({len(target_sizes)})"
        )
    
    return [
        upsample_feature_map(feat, size, mode=mode)
        for feat, size in zip(features, target_sizes)
    ]

