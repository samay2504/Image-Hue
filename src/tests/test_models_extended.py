"""
Extended model tests with memory safety for 6GB GPU and comprehensive coverage.
"""

import pytest
import torch
import torch.nn as nn

from src.models.model import (
    PaperNet, MobileLiteVariant, L2RegressionNet,
    get_model, count_parameters
)


class TestModelArchitecturesExtended:
    """Extended model architecture tests."""
    
    def test_papernet_with_gradient_checkpointing(self):
        """Test PaperNet with gradient checkpointing enabled."""
        model = PaperNet(num_classes=313, use_checkpointing=True)
        model.train()
        
        x = torch.randn(2, 1, 128, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert out.shape == (2, 313, 128, 128)
    
    def test_mobile_memory_efficient(self):
        """Test mobile variant with minimal memory."""
        # Very small channels for extreme memory constraint
        model = MobileLiteVariant(num_classes=313, base_channels=16)
        model.eval()
        
        x = torch.randn(1, 1, 256, 256)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (1, 313, 256, 256)
    
    def test_model_train_eval_modes(self):
        """Test train vs eval mode differences."""
        model = MobileLiteVariant(base_channels=32)
        
        x = torch.randn(2, 1, 128, 128)
        
        # Train mode
        model.train()
        out_train = model(x)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        
        # Shapes should match
        assert out_train.shape == out_eval.shape
    
    def test_different_num_classes(self):
        """Test models with different number of classes."""
        for num_classes in [100, 313, 500]:
            model = MobileLiteVariant(num_classes=num_classes, base_channels=16)
            x = torch.randn(1, 1, 64, 64)
            
            with torch.no_grad():
                out = model(x)
            
            assert out.shape == (1, num_classes, 64, 64)
    
    def test_l2_output_range(self):
        """Test L2 model output is in reasonable range."""
        model = L2RegressionNet(base_channels=32)
        model.eval()
        
        x = torch.randn(2, 1, 128, 128)
        
        with torch.no_grad():
            out = model(x)
        
        # Output should be ab values, roughly in [-128, 128]
        assert out.abs().max() < 200  # Allow some margin
    
    def test_model_determinism(self):
        """Test that eval mode is deterministic."""
        model = MobileLiteVariant(base_channels=16)
        model.eval()
        
        torch.manual_seed(42)
        x = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2)
    
    def test_model_on_different_devices(self):
        """Test model can move between devices."""
        model = MobileLiteVariant(base_channels=16)
        
        # CPU
        model = model.to('cpu')
        x_cpu = torch.randn(1, 1, 64, 64)
        out_cpu = model(x_cpu)
        assert out_cpu.device.type == 'cpu'
        
        # CUDA (if available)
        if torch.cuda.is_available():
            model = model.to('cuda')
            x_cuda = torch.randn(1, 1, 64, 64, device='cuda')
            out_cuda = model(x_cuda)
            assert out_cuda.device.type == 'cuda'
            
            # Clean up
            del model, x_cuda, out_cuda
            torch.cuda.empty_cache()


class TestModelFactoryExtended:
    """Extended tests for model factory."""
    
    def test_factory_with_all_options(self):
        """Test factory with various configuration options."""
        configs = [
            {'model_type': 'paper', 'num_classes': 313, 'use_checkpointing': True},
            {'model_type': 'mobile', 'num_classes': 313, 'base_channels': 32},
            {'model_type': 'mobile', 'num_classes': 313, 'base_channels': 16},
            {'model_type': 'l2', 'base_channels': 32},
            {'model_type': 'l2', 'base_channels': 16},
        ]
        
        for config in configs:
            model = get_model(config)
            assert model is not None
            assert isinstance(model, nn.Module)
    
    def test_factory_default_values(self):
        """Test factory with minimal config."""
        model = get_model({'model_type': 'mobile'})
        assert isinstance(model, MobileLiteVariant)
    
    def test_factory_invalid_type(self):
        """Test factory with invalid model type."""
        with pytest.raises((ValueError, KeyError)):
            get_model({'model_type': 'invalid_model'})


class TestParameterCounting:
    """Test parameter counting and memory estimation."""
    
    def test_parameter_count_accuracy(self):
        """Test parameter counting is accurate."""
        model = MobileLiteVariant(base_channels=16)
        
        # Manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Function count
        func_count = count_parameters(model)
        
        assert manual_count == func_count
    
    def test_parameter_count_different_models(self):
        """Test parameter counts for different models."""
        counts = {}
        
        configs = [
            ('paper', {'model_type': 'paper'}),
            ('mobile_32', {'model_type': 'mobile', 'base_channels': 32}),
            ('mobile_16', {'model_type': 'mobile', 'base_channels': 16}),
            ('l2_32', {'model_type': 'l2', 'base_channels': 32}),
        ]
        
        for name, config in configs:
            model = get_model(config)
            counts[name] = count_parameters(model)
        
        # Paper model should be largest
        assert counts['paper'] > counts['mobile_32']
        assert counts['mobile_32'] > counts['mobile_16']
    
    def test_trainable_vs_total_parameters(self):
        """Test counting only trainable parameters."""
        model = MobileLiteVariant(base_channels=16)
        
        # Freeze some layers
        for param in list(model.parameters())[:5]:
            param.requires_grad = False
        
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        
        assert trainable_count < total_count


class TestGradientFlow:
    """Test gradient flow through models."""
    
    def test_all_parameters_get_gradients(self):
        """Test that all trainable parameters receive gradients."""
        model = MobileLiteVariant(base_channels=16)
        model.train()
        
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
    
    def test_gradient_magnitude(self):
        """Test gradient magnitudes are reasonable."""
        model = MobileLiteVariant(base_channels=16)
        model.train()
        
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradients shouldn't be too large
                assert grad_norm < 100, f"Large gradient for {name}: {grad_norm}"


class TestMemorySafety:
    """Test memory safety for RTX 3060 6GB."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mobile_fits_in_6gb(self):
        """Test mobile model fits in 6GB GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Use mobile variant with moderate batch size
            model = MobileLiteVariant(num_classes=313, base_channels=32).cuda()
            model.eval()
            
            # Test batch size that should fit in 6GB
            batch_size = 4
            x = torch.randn(batch_size, 1, 256, 256, device='cuda')
            
            with torch.no_grad():
                out = model(x)
            
            assert out.shape == (batch_size, 313, 256, 256)
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            assert memory_allocated < 5.5, f"Too much memory used: {memory_allocated:.2f}GB"
            
            # Clean up
            del model, x, out
            torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_paper_model_with_checkpointing(self):
        """Test paper model with gradient checkpointing for 6GB."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            model = PaperNet(num_classes=313, use_checkpointing=True).cuda()
            model.train()
            
            # Small batch for training with checkpointing
            batch_size = 2
            x = torch.randn(batch_size, 1, 256, 256, device='cuda', requires_grad=True)
            
            out = model(x)
            loss = out.mean()
            loss.backward()
            
            # Check memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            assert memory_allocated < 5.5, f"Too much memory: {memory_allocated:.2f}GB"
            
            # Clean up
            del model, x, out, loss
            torch.cuda.empty_cache()
    
    def test_model_memory_estimate(self):
        """Estimate model memory footprint."""
        model = MobileLiteVariant(base_channels=32)
        
        # Calculate parameter memory (float32)
        param_count = count_parameters(model)
        param_memory_mb = (param_count * 4) / (1024**2)  # 4 bytes per float32
        
        print(f"\nModel parameters: {param_count:,}")
        print(f"Parameter memory: {param_memory_mb:.2f} MB")
        
        # Should be under 100MB for mobile variant
        assert param_memory_mb < 100, f"Model too large: {param_memory_mb:.2f}MB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp16_inference(self):
        """Test FP16 inference for memory savings."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            model = MobileLiteVariant(num_classes=313, base_channels=32).cuda().half()
            model.eval()
            
            x = torch.randn(4, 1, 256, 256, device='cuda', dtype=torch.float16)
            
            with torch.no_grad():
                out = model(x)
            
            assert out.dtype == torch.float16
            
            # FP16 should use roughly half the memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            assert memory_allocated < 3.0, f"FP16 using too much memory: {memory_allocated:.2f}GB"
            
            # Clean up
            del model, x, out
            torch.cuda.empty_cache()


class TestModelRobustness:
    """Test model robustness to edge cases."""
    
    def test_very_small_input(self):
        """Test with very small input size."""
        model = MobileLiteVariant(base_channels=16)
        model.eval()
        
        # 32x32 should still work
        x = torch.randn(1, 1, 32, 32)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape[2:] == (32, 32)
    
    def test_non_square_input(self):
        """Test with non-square input."""
        model = MobileLiteVariant(base_channels=16)
        model.eval()
        
        # 128x256 input
        x = torch.randn(1, 1, 128, 256)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape[2:] == (128, 256)
    
    def test_zero_input(self):
        """Test with zero input."""
        model = MobileLiteVariant(base_channels=16)
        model.eval()
        
        x = torch.zeros(1, 1, 64, 64)
        
        with torch.no_grad():
            out = model(x)
        
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))
    
    def test_extreme_input_values(self):
        """Test with extreme input values."""
        model = MobileLiteVariant(base_channels=16)
        model.eval()
        
        # Very large values
        x = torch.ones(1, 1, 64, 64) * 100
        
        with torch.no_grad():
            out = model(x)
        
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))


class TestModelSerialization:
    """Test model saving and loading."""
    
    def test_state_dict_save_load(self):
        """Test state dict save and load."""
        model1 = MobileLiteVariant(base_channels=16)
        
        # Save state
        state_dict = model1.state_dict()
        
        # Create new model and load state
        model2 = MobileLiteVariant(base_channels=16)
        model2.load_state_dict(state_dict)
        
        # Test they produce same output
        x = torch.randn(1, 1, 64, 64)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)
    
    def test_checkpoint_format(self):
        """Test checkpoint format compatibility."""
        model = MobileLiteVariant(base_channels=16)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {'model_type': 'mobile', 'base_channels': 16}
        }
        
        # Should be able to save/load
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save(checkpoint, temp_path)
            loaded = torch.load(temp_path)
            
            assert 'model_state_dict' in loaded
            assert 'config' in loaded
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
