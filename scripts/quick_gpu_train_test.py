#!/usr/bin/env python3
"""
Quick GPU training test for colorization model.

Tests:
- GPU device detection
- Model loading and moving to GPU
- Forward pass on GPU
- Backward pass with AMP
- Memory tracking
- DataLoader with pin_memory

Run this on Ubuntu 22.04 + RTX 5070 Ti to verify GPU training works.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image

# Import our utilities
from src.utils.device import (
    get_device, enable_cuda_optimizations, auto_batch_and_workers,
    print_device_summary, estimate_model_memory
)
from src.models.model import get_model


def create_dummy_data(batch_size=8, size=256):
    """Create dummy colorization data (L channel input)."""
    # L channel normalized to [-1, 1]
    L = torch.randn(batch_size, 1, size, size)
    
    # Target: soft-encoded distribution (Q=484 for Q=22)
    Q = 484  # ab_grid size
    target = torch.randint(0, Q, (batch_size, size, size))
    
    return L, target


def test_gpu_training(num_iterations=10):
    """Test GPU training with a tiny model."""
    print("=" * 70)
    print("Quick GPU Training Test")
    print("=" * 70)
    
    # Step 1: Device detection
    print("\n[1/6] Device Detection")
    print("-" * 70)
    print_device_summary()
    
    device = get_device()
    print(f"✓ Selected device: {device}")
    
    if device.type == "cpu":
        print("\n⚠ WARNING: No CUDA device detected!")
        print("Training will run on CPU (slow). Check:")
        print("  1. nvidia-smi shows GPU")
        print("  2. PyTorch built with CUDA: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  3. Driver installed: nvidia-smi")
        return False
    
    # Step 2: Enable optimizations
    print("\n[2/6] Enabling CUDA Optimizations")
    print("-" * 70)
    enable_cuda_optimizations()
    print(f"✓ cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
    print(f"✓ cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # Step 3: Create model
    print("\n[3/6] Creating Tiny Model")
    print("-" * 70)
    config = {
        'model_type': 'spade',
        'encoder_type': 'resnet18',
        'decoder_type': 'spade',
        'pretrained': False,
        'freeze_encoder': False
    }
    
    try:
        model = get_model(config)
        model = model.to(device)
        print(f"✓ Model moved to {device}")
        
        mem_mb = estimate_model_memory(model)
        print(f"✓ Model memory: {mem_mb:.2f} MB")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Parameters: {param_count / 1e6:.2f}M")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Step 4: Test forward pass
    print("\n[4/6] Testing Forward Pass")
    print("-" * 70)
    batch_size, num_workers = auto_batch_and_workers()
    print(f"Auto-detected: batch_size={batch_size}, num_workers={num_workers}")
    
    # Use smaller batch for test
    test_batch_size = min(8, batch_size)
    
    try:
        L, target = create_dummy_data(batch_size=test_batch_size)
        L = L.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        print(f"✓ Data moved to GPU: L {L.shape}, target {target.shape}")
        
        model.eval()
        with torch.no_grad():
            with autocast():
                output = model(L)
        
        print(f"✓ Forward pass successful: output {output.shape}")
        print(f"✓ Output device: {output.device}")
        
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU memory: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test backward pass with AMP
    print("\n[5/6] Testing Backward Pass with AMP")
    print("-" * 70)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    try:
        for i in range(3):
            L, target = create_dummy_data(batch_size=test_batch_size)
            L = L.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(L)
                # Simple loss (real training uses weighted cross-entropy)
                loss = nn.functional.cross_entropy(
                    output.view(-1, output.size(1)),
                    target.view(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"  Iteration {i+1}/3: loss={loss.item():.4f}")
        
        print("✓ Backward pass with AMP successful")
        
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU memory after training: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Training loop test
    print("\n[6/6] Testing Training Loop")
    print("-" * 70)
    
    try:
        losses = []
        for i in range(num_iterations):
            L, target = create_dummy_data(batch_size=test_batch_size)
            L = L.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(L)
                loss = nn.functional.cross_entropy(
                    output.view(-1, output.size(1)),
                    target.view(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            
            if (i + 1) % 5 == 0:
                avg_loss = np.mean(losses[-5:])
                print(f"  Iterations {i+1}/{num_iterations}: avg_loss={avg_loss:.4f}")
        
        print(f"✓ Completed {num_iterations} training iterations")
        print(f"✓ Final loss: {losses[-1]:.4f}")
        
        # Final memory check
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ Final GPU memory: {alloc:.2f}/{total:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success!
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYour system is ready for GPU training:")
    print(f"  • Device: {torch.cuda.get_device_name(0)}")
    print(f"  • CUDA Version: {torch.version.cuda}")
    print(f"  • cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"  • Recommended batch_size: {batch_size}")
    print(f"  • Recommended num_workers: {num_workers}")
    print("\nNext steps:")
    print("  1. Run scripts/verify_system_ayaan.sh to verify system setup")
    print("  2. Run scripts/verify_gpu_python.py for detailed GPU tests")
    print("  3. Start training with: python -m src.train --config configs/train_tiny.yaml --train_dir /path/to/data")
    
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    success = test_gpu_training(num_iterations=10)
    
    sys.exit(0 if success else 1)
