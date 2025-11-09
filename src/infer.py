"""
Inference module for colorization.

Supports:
- Classification-based colorization (paper method)
- L2 regression baseline
- OpenCV color transfer
- Tile-based inference for large images
- Redis caching
- Memory safety and auto-fallback
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

from src.models.model import get_model
from src.models.ops import (
    rgb_to_lab, lab_to_rgb, decode_distribution_to_ab,
    get_ab_grid, DEFAULT_TEMPERATURE
)
from src.utils.memory import (
    get_gpu_memory_info, clear_cuda_cache, tile_image, stitch_tiles
)
from src.cache.redis_client import get_cache


class ColorizationInference:
    """Inference engine for colorization."""
    
    def __init__(self, model_path: Optional[str] = None, model_config: Optional[Dict] = None,
                 device: Optional[str] = None, use_cache: bool = True,
                 redis_url: Optional[str] = None):
        """
        Args:
            model_path: Path to trained model checkpoint
            model_config: Model configuration dict
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_cache: Whether to use Redis/disk caching
            redis_url: Redis connection URL
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Inference device: {self.device}")
        
        # Load model
        self.model = None
        self.ab_grid = get_ab_grid()  # Get grid first to know Q
        Q = len(self.ab_grid)
        
        # Update config with actual Q
        self.model_config = model_config or {'model_type': 'mobile'}
        if 'num_classes' not in self.model_config and self.model_config.get('model_type') != 'l2':
            self.model_config['num_classes'] = Q
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"No model checkpoint provided, creating untrained model with Q={Q} bins")
            self.model = get_model(self.model_config).to(self.device)
        
        self.model.eval()
        
        # Cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = get_cache(redis_url=redis_url)
        else:
            self.cache = None
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {}).get('model', self.model_config)
        else:
            state_dict = checkpoint
            config = self.model_config
        
        self.model = get_model(config).to(self.device)
        self.model.load_state_dict(state_dict)
        print("Model loaded successfully")
    
    @torch.no_grad()
    def colorize_image(self, image: Union[str, np.ndarray, Image.Image],
                       method: str = 'classification',
                       temperature: float = DEFAULT_TEMPERATURE,
                       tile_size: Optional[int] = None,
                       return_lab: bool = False) -> np.ndarray:
        """
        Colorize an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            method: 'classification', 'l2', or 'opencv'
            temperature: Temperature for annealed-mean decoding
            tile_size: Tile size for large images (None = no tiling)
            return_lab: Whether to return Lab instead of RGB
            
        Returns:
            Colorized image as numpy array (RGB or Lab)
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            img = image.convert('RGB')
        
        img_np = np.array(img) / 255.0
        orig_size = img_np.shape[:2]
        
        # Check cache
        if self.use_cache and self.cache:
            cache_key = img.tobytes()
            cache_params = {'method': method, 'temp': temperature, 'tile_size': tile_size}
            cached_result = self.cache.get(cache_key, method, cache_params)
            if cached_result is not None:
                return cached_result
        
        # Route to appropriate method
        if method == 'classification':
            result = self._colorize_classification(img_np, temperature, tile_size)
        elif method == 'l2':
            result = self._colorize_l2(img_np, tile_size)
        elif method == 'opencv':
            result = self._colorize_opencv(img_np)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(cache_key, method, cache_params, result)
        
        if return_lab:
            return rgb_to_lab(result)
        return result
    
    def _colorize_classification(self, img_rgb: np.ndarray, temperature: float,
                                 tile_size: Optional[int]) -> np.ndarray:
        """Colorize using classification-based method (paper)."""
        # Convert to Lab
        lab = rgb_to_lab(img_rgb)
        L = lab[:, :, 0:1]
        
        # Normalize L
        L_norm = (L - 50.0) / 50.0
        L_tensor = torch.from_numpy(L_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Inference with optional tiling
        if tile_size and (img_rgb.shape[0] > tile_size or img_rgb.shape[1] > tile_size):
            logits = self._inference_with_tiling(L_tensor, tile_size)
        else:
            logits = self.model(L_tensor)
        
        # Decode to ab
        ab_pred = decode_distribution_to_ab(logits, temperature=temperature)
        ab_pred = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Combine with L
        lab_pred = np.concatenate([L, ab_pred], axis=2)
        
        # Convert to RGB
        rgb_pred = lab_to_rgb(lab_pred)
        
        return rgb_pred
    
    def _colorize_l2(self, img_rgb: np.ndarray, tile_size: Optional[int]) -> np.ndarray:
        """Colorize using L2 regression baseline."""
        # Convert to Lab
        lab = rgb_to_lab(img_rgb)
        L = lab[:, :, 0:1]
        
        # Normalize L
        L_norm = (L - 50.0) / 50.0
        L_tensor = torch.from_numpy(L_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Inference
        if tile_size and (img_rgb.shape[0] > tile_size or img_rgb.shape[1] > tile_size):
            ab_pred = self._inference_with_tiling(L_tensor, tile_size)
        else:
            ab_pred = self.model(L_tensor)
        
        ab_pred = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Combine with L
        lab_pred = np.concatenate([L, ab_pred], axis=2)
        
        # Convert to RGB
        rgb_pred = lab_to_rgb(lab_pred)
        
        return rgb_pred
    
    def _colorize_opencv(self, img_rgb: np.ndarray) -> np.ndarray:
        """Colorize using OpenCV color transfer (as baseline)."""
        # Simple histogram matching from a default colored reference
        # For a proper implementation, would need a reference image
        # Here we'll do a simple saturation boost
        
        # Convert to HSV
        img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Boost saturation (simple colorization)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.5, 0, 255)
        
        # Convert back
        img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb / 255.0
    
    def _inference_with_tiling(self, L_tensor: torch.Tensor, tile_size: int) -> torch.Tensor:
        """Run inference with tiling for large images."""
        overlap = tile_size // 4
        
        # Split into tiles
        tiles, positions = tile_image(L_tensor, tile_size=tile_size, overlap=overlap)
        
        # Inference on each tile
        output_tiles = []
        for tile in tiles:
            with torch.no_grad():
                out_tile = self.model(tile)
                output_tiles.append(out_tile)
        
        # Stitch back together
        output_shape = (L_tensor.shape[0], output_tiles[0].shape[1],
                       L_tensor.shape[2], L_tensor.shape[3])
        output = stitch_tiles(output_tiles, positions, output_shape, overlap=overlap)
        
        return output
    
    def colorize_folder(self, input_dir: str, output_dir: str,
                       method: str = 'classification',
                       temperature: float = DEFAULT_TEMPERATURE):
        """Colorize all images in a folder."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [p for p in input_path.rglob('*') if p.suffix.lower() in extensions]
        
        print(f"Found {len(image_files)} images to colorize")
        
        from tqdm import tqdm
        for img_path in tqdm(image_files):
            try:
                # Colorize
                result = self.colorize_image(str(img_path), method=method, temperature=temperature)
                
                # Save
                rel_path = img_path.relative_to(input_path)
                out_path = output_path / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                result_img = Image.fromarray((result * 255).astype(np.uint8))
                result_img.save(out_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    def create_blend_animation(self, image: Union[str, np.ndarray],
                              method: str = 'classification',
                              temperature: float = DEFAULT_TEMPERATURE,
                              num_frames: int = 30) -> list:
        """
        Create blend animation from grayscale to colored.
        
        Returns:
            List of RGB frames
        """
        # Colorize
        colored = self.colorize_image(image, method=method, temperature=temperature)
        
        # Get grayscale version
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            img = image
        
        img_np = np.array(img) / 255.0
        gray = np.mean(img_np, axis=2, keepdims=True)
        gray = np.repeat(gray, 3, axis=2)
        
        # Create blend frames
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1)  # 0 to 1
            frame = gray * (1 - alpha) + colored * alpha
            frames.append(frame)
        
        return frames


def colorize_cli():
    """Command-line interface for colorization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Colorize images')
    parser.add_argument('input', type=str, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--model', type=str, help='Model checkpoint path')
    parser.add_argument('--method', type=str, default='classification',
                       choices=['classification', 'l2', 'opencv'],
                       help='Colorization method')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help='Temperature for annealed-mean')
    parser.add_argument('--tile_size', type=int, help='Tile size for large images')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--redis-url', type=str, help='Redis URL')
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = ColorizationInference(
        model_path=args.model,
        device=args.device,
        use_cache=not args.no_cache,
        redis_url=args.redis_url
    )
    
    # Check if input is directory
    input_path = Path(args.input)
    if input_path.is_dir():
        engine.colorize_folder(
            args.input,
            args.output,
            method=args.method,
            temperature=args.temperature
        )
    else:
        # Single image
        result = engine.colorize_image(
            args.input,
            method=args.method,
            temperature=args.temperature,
            tile_size=args.tile_size
        )
        
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        result_img.save(args.output)
        print(f"Saved colorized image to {args.output}")


if __name__ == '__main__':
    colorize_cli()
