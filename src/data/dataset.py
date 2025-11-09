"""
Dataset classes for colorization training and evaluation.
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ColorizationDataset(Dataset):
    """
    Dataset for colorization training.
    
    Loads color images and converts to Lab space, returning L as input
    and ab as target.
    """
    
    def __init__(self, image_dir: str, image_paths: Optional[List[str]] = None,
                 transform: Optional[Callable] = None, target_size: int = 256,
                 use_soft_encoding: bool = True):
        """
        Args:
            image_dir: Directory containing images
            image_paths: Optional list of image paths (relative to image_dir)
            transform: Optional transform for data augmentation
            target_size: Target size for images
            use_soft_encoding: Whether to use soft-encoding for targets
        """
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.use_soft_encoding = use_soft_encoding
        
        # Find all images if paths not provided
        if image_paths is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            self.image_paths = [
                str(p.relative_to(self.image_dir))
                for p in self.image_dir.rglob('*')
                if p.suffix.lower() in extensions
            ]
        else:
            self.image_paths = image_paths
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            L: (1, H, W) lightness channel
            ab: (2, H, W) ab channels (if not using soft-encoding)
            target: (Q, H, W) soft-encoded distribution or (2, H, W) ab channels
        """
        # Load image
        img_path = self.image_dir / self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image
            img = Image.new('RGB', (self.target_size, self.target_size), (128, 128, 128))
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((self.target_size, self.target_size))
        
        # Convert to numpy
        img_np = np.array(img) / 255.0
        
        # Convert to Lab
        from src.models.ops import rgb_to_lab
        lab = rgb_to_lab(img_np)
        
        # Split into L and ab
        L = lab[:, :, 0:1]  # (H, W, 1)
        ab = lab[:, :, 1:]  # (H, W, 2)
        
        # Normalize L to [-1, 1] (L is in [0, 100])
        L = (L - 50.0) / 50.0
        
        # Convert to tensors (C, H, W)
        L = torch.from_numpy(L).permute(2, 0, 1).float()
        ab = torch.from_numpy(ab).permute(2, 0, 1).float()
        
        # Soft-encode target if requested
        if self.use_soft_encoding:
            from src.models.ops import encode_ab_to_distribution
            ab_np = ab.permute(1, 2, 0).numpy()  # (H, W, 2)
            target_dist = encode_ab_to_distribution(ab_np)  # (H, W, Q)
            target = torch.from_numpy(target_dist).permute(2, 0, 1).float()  # (Q, H, W)
        else:
            target = ab
        
        return L, ab, target


class ImageFolderDataset(Dataset):
    """Simple dataset for inference - just loads and preprocesses images."""
    
    def __init__(self, image_dir: str, target_size: int = 256):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_paths = [
            p for p in self.image_dir.rglob('*')
            if p.suffix.lower() in extensions
        ]
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            L: (1, H, W) lightness channel tensor
            path: Image path
        """
        img_path = self.image_paths[idx]
        
        # Load and convert to RGB
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size
        
        # Resize
        img = img.resize((self.target_size, self.target_size), Image.BICUBIC)
        
        # Convert to Lab
        img_np = np.array(img) / 255.0
        from src.models.ops import rgb_to_lab
        lab = rgb_to_lab(img_np)
        
        L = lab[:, :, 0:1]
        L = (L - 50.0) / 50.0  # Normalize
        L = torch.from_numpy(L).permute(2, 0, 1).float()
        
        return L, str(img_path)


def create_data_loaders(train_dir: str, val_dir: Optional[str] = None,
                        batch_size: int = 16, num_workers: int = 4,
                        target_size: int = 256) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create train and validation data loaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        target_size: Image target size
        
    Returns:
        train_loader, val_loader (val_loader is None if val_dir not provided)
    """
    # Training dataset
    train_dataset = ColorizationDataset(
        train_dir,
        target_size=target_size,
        use_soft_encoding=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset
    val_loader = None
    if val_dir:
        val_dataset = ColorizationDataset(
            val_dir,
            target_size=target_size,
            use_soft_encoding=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    return train_loader, val_loader


def compute_dataset_color_statistics(data_dir: str, output_file: str,
                                      max_images: int = 10000):
    """
    Compute color distribution statistics from dataset.
    
    Args:
        data_dir: Directory containing color images
        output_file: Path to save statistics (numpy .npz file)
        max_images: Maximum number of images to process
    """
    from src.models.ops import compute_empirical_distribution_from_images, get_ab_grid
    
    # Find all images
    data_path = Path(data_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        str(p) for p in data_path.rglob('*')
        if p.suffix.lower() in extensions
    ][:max_images]
    
    print(f"Computing color statistics from {len(image_paths)} images...")
    
    # Compute empirical distribution
    empirical_dist = compute_empirical_distribution_from_images(image_paths)
    
    # Compute rebalancing weights
    from src.models.ops import compute_class_rebalancing_weights
    weights = compute_class_rebalancing_weights(empirical_dist)
    
    # Save
    ab_grid = get_ab_grid()
    np.savez(output_file,
             empirical_distribution=empirical_dist,
             class_weights=weights,
             ab_grid=ab_grid)
    
    print(f"Saved color statistics to {output_file}")
    print(f"Empirical distribution shape: {empirical_dist.shape}")
    print(f"Class weights shape: {weights.shape}")
    print(f"Class weights - min: {weights.min():.3f}, max: {weights.max():.3f}, mean: {weights.mean():.3f}")


if __name__ == '__main__':
    # Test dataset loading
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        
        print("Testing ColorizationDataset...")
        dataset = ColorizationDataset(data_dir, target_size=256)
        
        if len(dataset) > 0:
            L, ab, target = dataset[0]
            print(f"L shape: {L.shape}, range: [{L.min():.2f}, {L.max():.2f}]")
            print(f"ab shape: {ab.shape}, range: [{ab.min():.2f}, {ab.max():.2f}]")
            print(f"Target shape: {target.shape}")
            
            # Test dataloader
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
            for batch in loader:
                L_batch, ab_batch, target_batch = batch
                print(f"\nBatch L: {L_batch.shape}")
                print(f"Batch ab: {ab_batch.shape}")
                print(f"Batch target: {target_batch.shape}")
                break
    else:
        print("Usage: python dataset.py <data_directory>")
