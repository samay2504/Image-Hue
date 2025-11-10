#!/usr/bin/env python3
"""
GPU verification script for PyTorch training setup.
Tests CUDA availability, device detection, and basic forward/backward pass.

Usage: python3 scripts/verify_gpu_python.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def verify_cuda_basics():
    """Verify basic CUDA availability and device info."""
    print_section("CUDA Basic Information")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Compiled Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ ERROR: CUDA is not available!")
        print("Possible causes:")
        print("  1. NVIDIA drivers not installed")
        print("  2. PyTorch installed without CUDA support")
        print("  3. CUDA_VISIBLE_DEVICES set incorrectly")
        return False
    
    print("✓ CUDA is available")
    
    device_count = torch.cuda.device_count()
    print(f"\nGPU Count: {device_count}")
    
    for i in range(device_count):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    return True


def verify_memory_management():
    """Verify CUDA memory management."""
    print_section("CUDA Memory Management")
    
    if not torch.cuda.is_available():
        print("Skipping (CUDA not available)")
        return False
    
    device = torch.device('cuda:0')
    
    # Before allocation
    allocated_before = torch.cuda.memory_allocated(device) / 1e6
    reserved_before = torch.cuda.memory_reserved(device) / 1e6
    
    print(f"Memory allocated (before): {allocated_before:.2f} MB")
    print(f"Memory reserved (before): {reserved_before:.2f} MB")
    
    # Allocate tensor
    test_tensor = torch.randn(1000, 1000, device=device)
    
    allocated_after = torch.cuda.memory_allocated(device) / 1e6
    reserved_after = torch.cuda.memory_reserved(device) / 1e6
    
    print(f"\nMemory allocated (after 1000x1000 tensor): {allocated_after:.2f} MB")
    print(f"Memory reserved (after): {reserved_after:.2f} MB")
    print(f"Difference: {allocated_after - allocated_before:.2f} MB")
    
    # Cleanup
    del test_tensor
    torch.cuda.empty_cache()
    
    print("\n✓ Memory management working")
    return True


def verify_simple_model():
    """Verify simple model can run on GPU."""
    print_section("Simple Model GPU Test")
    
    if not torch.cuda.is_available():
        print("Skipping (CUDA not available)")
        return False
    
    device = torch.device('cuda:0')
    
    # Define simple conv model
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    print("Creating model...")
    model = SimpleConvNet()
    
    print(f"Model parameters (before .to(device)): {next(model.parameters()).device}")
    
    # Move to GPU
    model = model.to(device)
    print(f"Model parameters (after .to(device)): {next(model.parameters()).device}")
    
    # Create random input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32, device=device)
    print(f"\nInput tensor device: {input_tensor.device}")
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    output = model(input_tensor)
    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")
    
    if output.device.type != 'cuda':
        print("\n❌ ERROR: Output not on CUDA device!")
        return False
    
    print("\n✓ Forward pass on GPU successful")
    return True


def verify_amp_training():
    """Verify AMP (Automatic Mixed Precision) training."""
    print_section("AMP Training Test")
    
    if not torch.cuda.is_available():
        print("Skipping (CUDA not available)")
        return False
    
    device = torch.device('cuda:0')
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Dummy data
    inputs = torch.randn(32, 128, device=device)
    targets = torch.randint(0, 10, (32,), device=device)
    
    print("Running training step with AMP...")
    
    mem_before = torch.cuda.memory_allocated(device) / 1e6
    
    # Training step with AMP
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    mem_after = torch.cuda.memory_allocated(device) / 1e6
    mem_peak = torch.cuda.max_memory_allocated(device) / 1e6
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    print(f"Memory peak: {mem_peak:.2f} MB")
    
    # Verify gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    if not has_grad:
        print("\n❌ ERROR: No gradients computed!")
        return False
    
    print("\n✓ AMP training step successful")
    return True


def verify_cudnn():
    """Verify cuDNN configuration."""
    print_section("cuDNN Configuration")
    
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    if not torch.backends.cudnn.enabled:
        print("\n⚠ WARNING: cuDNN is disabled (may impact performance)")
    
    print("\n✓ cuDNN configuration checked")
    return True


def main():
    """Run all verification tests."""
    print("="*60)
    print("  GPU Verification Script for PyTorch Training")
    print("  Ubuntu 22.04 + RTX 5070 Ti + cu128")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("CUDA Basics", verify_cuda_basics()))
    results.append(("Memory Management", verify_memory_management()))
    results.append(("Simple Model", verify_simple_model()))
    results.append(("AMP Training", verify_amp_training()))
    results.append(("cuDNN Config", verify_cudnn()))
    
    # Summary
    print_section("Verification Summary")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - GPU is ready for training!")
        print("="*60)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please review errors above")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
