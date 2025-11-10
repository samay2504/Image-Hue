#!/usr/bin/env python3
"""
DataLoader profiling script for RTX 5070 Ti + Ryzen 7 9700X.

Benchmarks different num_workers values to find optimal setting.
Tests pin_memory, persistent_workers, prefetch_factor impact.

Run on Ubuntu 22.04 with actual training data to profile real I/O patterns.
"""

import sys
import os
import time
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import ColorizationDataset
from src.utils.device import get_device, print_device_summary


def benchmark_dataloader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
    num_batches=50
):
    """
    Benchmark DataLoader throughput.
    
    Returns:
        batches_per_sec, samples_per_sec
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True
    )
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    
    # Benchmark
    start = time.time()
    batches_loaded = 0
    
    for i, (L, ab, target) in enumerate(loader):
        batches_loaded += 1
        if batches_loaded >= num_batches:
            break
    
    elapsed = time.time() - start
    
    batches_per_sec = batches_loaded / elapsed
    samples_per_sec = (batches_loaded * batch_size) / elapsed
    
    return batches_per_sec, samples_per_sec


def profile_num_workers(data_dir, batch_size=16, num_batches=50):
    """Profile different num_workers values."""
    print("=" * 80)
    print("DataLoader Profiling: num_workers")
    print("=" * 80)
    
    dataset = ColorizationDataset(data_dir, target_size=256)
    print(f"\nDataset: {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Benchmarking {num_batches} batches per configuration\n")
    
    # Test different num_workers values
    cpu_count = os.cpu_count() or 1
    test_workers = [0, 2, 4, 6, 8, 10, 12, 16, min(cpu_count, 20)]
    test_workers = sorted(set(test_workers))  # Remove duplicates
    
    results = []
    
    print("-" * 80)
    print(f"{'num_workers':<12} {'batches/sec':<15} {'samples/sec':<15} {'speedup':<10}")
    print("-" * 80)
    
    baseline_samples_per_sec = None
    
    for workers in test_workers:
        try:
            batches_ps, samples_ps = benchmark_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
                prefetch_factor=2,
                num_batches=num_batches
            )
            
            if baseline_samples_per_sec is None:
                baseline_samples_per_sec = samples_ps
                speedup = 1.0
            else:
                speedup = samples_ps / baseline_samples_per_sec
            
            results.append({
                'workers': workers,
                'batches_ps': batches_ps,
                'samples_ps': samples_ps,
                'speedup': speedup
            })
            
            print(f"{workers:<12} {batches_ps:<15.2f} {samples_ps:<15.1f} {speedup:<10.2f}x")
        
        except Exception as e:
            print(f"{workers:<12} ERROR: {e}")
    
    print("-" * 80)
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x['samples_ps'])
        print(f"\n✓ Best configuration: num_workers={best['workers']}")
        print(f"  Throughput: {best['samples_ps']:.1f} samples/sec ({best['speedup']:.2f}x speedup)")
        print(f"  Batches/sec: {best['batches_ps']:.2f}")
        
        return best['workers']
    
    return 4


def profile_configurations(data_dir, batch_size=16, num_workers=8, num_batches=50):
    """Profile different DataLoader configurations."""
    print("\n" + "=" * 80)
    print("DataLoader Profiling: Configurations")
    print("=" * 80)
    
    dataset = ColorizationDataset(data_dir, target_size=256)
    print(f"\nDataset: {len(dataset)} images")
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print(f"Benchmarking {num_batches} batches per configuration\n")
    
    configs = [
        {"name": "Baseline", "pin_memory": False, "persistent": False, "prefetch": None},
        {"name": "pin_memory", "pin_memory": True, "persistent": False, "prefetch": None},
        {"name": "pin + persistent", "pin_memory": True, "persistent": True, "prefetch": None},
        {"name": "pin + persistent + prefetch=2", "pin_memory": True, "persistent": True, "prefetch": 2},
        {"name": "pin + persistent + prefetch=4", "pin_memory": True, "persistent": True, "prefetch": 4},
    ]
    
    results = []
    
    print("-" * 80)
    print(f"{'Configuration':<35} {'batches/sec':<15} {'samples/sec':<15} {'speedup':<10}")
    print("-" * 80)
    
    baseline_samples_per_sec = None
    
    for config in configs:
        try:
            batches_ps, samples_ps = benchmark_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers if num_workers > 0 else 0,
                pin_memory=config['pin_memory'] and torch.cuda.is_available(),
                persistent_workers=config['persistent'] and num_workers > 0,
                prefetch_factor=config['prefetch'],
                num_batches=num_batches
            )
            
            if baseline_samples_per_sec is None:
                baseline_samples_per_sec = samples_ps
                speedup = 1.0
            else:
                speedup = samples_ps / baseline_samples_per_sec
            
            results.append({
                'name': config['name'],
                'batches_ps': batches_ps,
                'samples_ps': samples_ps,
                'speedup': speedup
            })
            
            print(f"{config['name']:<35} {batches_ps:<15.2f} {samples_ps:<15.1f} {speedup:<10.2f}x")
        
        except Exception as e:
            print(f"{config['name']:<35} ERROR: {e}")
    
    print("-" * 80)
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x['samples_ps'])
        print(f"\n✓ Best configuration: {best['name']}")
        print(f"  Throughput: {best['samples_ps']:.1f} samples/sec ({best['speedup']:.2f}x speedup)")
        print(f"  Batches/sec: {best['batches_ps']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Profile DataLoader performance')
    parser.add_argument('data_dir', type=str, help='Directory containing training images')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--num-batches', type=int, default=50, help='Number of batches to benchmark')
    parser.add_argument('--skip-workers', action='store_true', help='Skip num_workers profiling')
    parser.add_argument('--skip-configs', action='store_true', help='Skip configuration profiling')
    args = parser.parse_args()
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        sys.exit(1)
    
    # Print device info
    print_device_summary()
    
    # Profile num_workers
    best_workers = 8
    if not args.skip_workers:
        best_workers = profile_num_workers(
            args.data_dir,
            batch_size=args.batch_size,
            num_batches=args.num_batches
        )
    
    # Profile configurations
    if not args.skip_configs:
        profile_configurations(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=best_workers,
            num_batches=args.num_batches
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("Profiling Summary")
    print("=" * 80)
    print(f"\nRecommended settings for your system:")
    print(f"  num_workers: {best_workers}")
    print(f"  pin_memory: {torch.cuda.is_available()}")
    print(f"  persistent_workers: True")
    print(f"  prefetch_factor: 2")
    print("\nUpdate your training config:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  num_workers: {best_workers}")
    print("\nNote: Optimal num_workers depends on:")
    print("  - CPU core count (faster CPUs = more workers)")
    print("  - Storage speed (SSD vs HDD)")
    print("  - Image preprocessing complexity")
    print("  - GPU training speed (faster GPU = more data needed)")
    print("\nFor RTX 5070 Ti (12GB) + Ryzen 7 9700X (16 cores):")
    print("  Typical range: 8-12 workers for balanced CPU/GPU utilization")


if __name__ == "__main__":
    main()
