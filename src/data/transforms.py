"""Data augmentation transforms for colorization."""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from typing import Tuple
from PIL import Image


class RandomResizedCropAndFlip:
    """Random resized crop with optional horizontal flip."""
    
    def __init__(self, size: int = 256, scale: Tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # Random resized crop
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=self.scale, ratio=(0.9, 1.1)
        )
        img = TF.resized_crop(img, i, j, h, w, (self.size, self.size))
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        return img


def get_train_transforms(size: int = 256) -> T.Compose:
    """Get training data augmentation pipeline."""
    return RandomResizedCropAndFlip(size=size)


def get_val_transforms(size: int = 256) -> T.Compose:
    """Get validation transform (just resize)."""
    return T.Resize((size, size))
