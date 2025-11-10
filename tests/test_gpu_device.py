"""
Unit tests for GPU device detection and training utilities.

Run with: pytest tests/test_gpu_device.py -v
Skip if no GPU: pytest tests/test_gpu_device.py -v -k "not gpu"
"""

import pytest
import torch
import torch.nn as nn

from src.utils.device import (
    get_device, get_device_info, auto_batch_and_workers,
    enable_cuda_optimizations, get_dataloader_config, estimate_model_memory
)
from src.utils.oom_protection import OOMHandler


# Skip tests if CUDA not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestDeviceDetection:
    """Test device detection utilities."""
    
    def test_get_device_returns_device(self):
        """Test get_device returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "cpu"]
    
    def test_get_device_prefers_cuda(self):
        """Test get_device prefers CUDA when available."""
        device = get_device(prefer="cuda")
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"
    
    def test_get_device_can_force_cpu(self):
        """Test get_device can force CPU."""
        device = get_device(prefer="cpu")
        assert device.type == "cpu"
    
    def test_get_device_info_structure(self):
        """Test get_device_info returns expected structure."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "device_count" in info
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["device_count"], int)
    
    @requires_cuda
    def test_get_device_info_cuda_details(self):
        """Test get_device_info returns CUDA details when available."""
        info = get_device_info()
        assert info["cuda_available"] is True
        assert info["device_count"] > 0
        assert "cuda_version" in info
        assert "cudnn_version" in info
        assert "gpu_0" in info
        
        gpu_info = info["gpu_0"]
        assert "name" in gpu_info
        assert "total_memory_gb" in gpu_info
        assert "capability" in gpu_info
        assert gpu_info["total_memory_gb"] > 0


class TestBatchAndWorkers:
    """Test batch size and num_workers auto-configuration."""
    
    def test_auto_batch_and_workers_returns_tuple(self):
        """Test auto_batch_and_workers returns (batch_size, num_workers)."""
        batch_size, num_workers = auto_batch_and_workers()
        assert isinstance(batch_size, int)
        assert isinstance(num_workers, int)
        assert batch_size > 0
        assert num_workers >= 0
    
    def test_auto_batch_and_workers_respects_image_size(self):
        """Test larger images get smaller batch sizes."""
        batch_small, _ = auto_batch_and_workers(image_size=128)
        batch_large, _ = auto_batch_and_workers(image_size=512)
        assert batch_large <= batch_small
    
    @requires_cuda
    def test_auto_batch_and_workers_cuda_larger_than_cpu(self):
        """Test CUDA gets larger batch sizes than CPU."""
        # Force CPU
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_cuda, _ = auto_batch_and_workers(vram_gb=vram_gb)
        batch_cpu, _ = auto_batch_and_workers(vram_gb=0)  # Simulate CPU
        assert batch_cuda >= batch_cpu
    
    def test_get_dataloader_config_structure(self):
        """Test get_dataloader_config returns valid dict."""
        config = get_dataloader_config()
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "num_workers" in config
        assert "pin_memory" in config
        assert "persistent_workers" in config
        assert "prefetch_factor" in config
        
        assert isinstance(config["batch_size"], int)
        assert isinstance(config["num_workers"], int)
        assert isinstance(config["pin_memory"], bool)
        assert isinstance(config["persistent_workers"], bool)
    
    @requires_cuda
    def test_get_dataloader_config_enables_pin_memory_on_cuda(self):
        """Test pin_memory is True when CUDA available."""
        config = get_dataloader_config()
        assert config["pin_memory"] is True


class TestCUDAOptimizations:
    """Test CUDA optimization utilities."""
    
    def test_enable_cuda_optimizations_no_error(self):
        """Test enable_cuda_optimizations runs without error."""
        # Should not raise even without CUDA
        enable_cuda_optimizations()
    
    @requires_cuda
    def test_enable_cuda_optimizations_enables_cudnn(self):
        """Test enable_cuda_optimizations enables cuDNN benchmark."""
        enable_cuda_optimizations()
        assert torch.backends.cudnn.enabled is True
        assert torch.backends.cudnn.benchmark is True


class TestModelMemory:
    """Test model memory estimation."""
    
    def test_estimate_model_memory(self):
        """Test estimate_model_memory calculates size correctly."""
        # Simple model
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
        )
        
        mem_mb = estimate_model_memory(model)
        assert isinstance(mem_mb, float)
        assert mem_mb > 0
        assert mem_mb < 1  # Should be less than 1MB for this tiny model


@requires_cuda
class TestGPUTraining:
    """Test GPU training functionality (requires CUDA)."""
    
    def test_model_moves_to_gpu(self):
        """Test model can be moved to GPU."""
        device = get_device()
        assert device.type == "cuda"
        
        model = nn.Sequential(nn.Linear(10, 10))
        model = model.to(device)
        
        # Check model is on GPU
        assert next(model.parameters()).device.type == "cuda"
    
    def test_tensor_moves_to_gpu(self):
        """Test tensors can be moved to GPU."""
        device = get_device()
        assert device.type == "cuda"
        
        tensor = torch.randn(10, 10)
        tensor = tensor.to(device)
        
        assert tensor.device.type == "cuda"
    
    def test_forward_pass_on_gpu(self):
        """Test forward pass runs on GPU."""
        device = get_device()
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
        ).to(device)
        
        x = torch.randn(2, 1, 32, 32).to(device)
        output = model(x)
        
        assert output.device.type == "cuda"
        assert output.shape == (2, 16, 30, 30)
    
    def test_backward_pass_on_gpu(self):
        """Test backward pass runs on GPU."""
        device = get_device()
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(2, 1, 32, 32).to(device)
        target = torch.randn(2, 1).to(device)
        
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check gradients computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.device.type == "cuda"
    
    def test_amp_training_on_gpu(self):
        """Test AMP training works on GPU."""
        from torch.cuda.amp import autocast, GradScaler
        
        device = get_device()
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
        
        x = torch.randn(2, 1, 32, 32).to(device)
        target = torch.randn(2, 1).to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check training step completed
        assert loss.item() > 0


class TestOOMHandler:
    """Test OOM protection utilities."""
    
    def test_oom_handler_init(self):
        """Test OOMHandler initializes correctly."""
        handler = OOMHandler(initial_batch_size=32, min_batch_size=4)
        assert handler.current_batch_size == 32
        assert handler.min_batch_size == 4
        assert handler.retry_count == 0
    
    def test_is_oom_error_detects_oom(self):
        """Test is_oom_error detects OOM exceptions."""
        oom_error = RuntimeError("CUDA out of memory. Tried to allocate...")
        assert OOMHandler.is_oom_error(oom_error) is True
        
        other_error = RuntimeError("Something else")
        assert OOMHandler.is_oom_error(other_error) is False
        
        value_error = ValueError("Not a runtime error")
        assert OOMHandler.is_oom_error(value_error) is False
    
    def test_on_oom_reduces_batch_size(self):
        """Test on_oom reduces batch size."""
        handler = OOMHandler(initial_batch_size=32, min_batch_size=4, reduction_factor=0.5)
        
        initial_batch = handler.current_batch_size
        handler.on_oom()
        
        assert handler.current_batch_size < initial_batch
        assert handler.current_batch_size == 16  # 32 * 0.5
        assert handler.retry_count == 1
    
    def test_on_oom_increases_grad_accum_steps(self):
        """Test on_oom increases gradient accumulation steps."""
        handler = OOMHandler(
            initial_batch_size=32,
            min_batch_size=4,
            reduction_factor=0.5,
            enable_grad_accumulation=True
        )
        
        assert handler.grad_accum_steps == 1
        handler.on_oom()
        assert handler.grad_accum_steps == 2  # Maintains effective batch size
    
    def test_on_oom_fails_at_max_retries(self):
        """Test on_oom raises error at max retries."""
        handler = OOMHandler(initial_batch_size=32, min_batch_size=4, max_retries=2)
        
        handler.on_oom()  # Retry 1
        handler.on_oom()  # Retry 2
        
        with pytest.raises(RuntimeError, match="max retries"):
            handler.on_oom()  # Should raise
    
    def test_on_oom_fails_at_min_batch_size(self):
        """Test on_oom raises error when cannot reduce further."""
        handler = OOMHandler(initial_batch_size=4, min_batch_size=4)
        
        with pytest.raises(RuntimeError, match="minimum"):
            handler.on_oom()
    
    def test_get_current_config(self):
        """Test get_current_config returns correct values."""
        handler = OOMHandler(initial_batch_size=32)
        config = handler.get_current_config()
        
        assert config['batch_size'] == 32
        assert config['grad_accum_steps'] == 1
        assert config['effective_batch_size'] == 32
        
        handler.on_oom()
        config = handler.get_current_config()
        assert config['batch_size'] == 16
        assert config['effective_batch_size'] == 32  # Maintained via grad_accum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
