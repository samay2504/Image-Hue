"""
Unit tests for model architectures.
"""

import pytest
import torch

from src.models.model import (
    PaperNet, MobileLiteVariant, L2RegressionNet,
    get_model, count_parameters
)


class TestModelArchitectures:
    """Test model forward passes and shapes."""
    
    def test_papernet_forward(self):
        """Test PaperNet forward pass."""
        model = PaperNet(num_classes=313)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (2, 313, 256, 256)
    
    def test_mobile_variant_forward(self):
        """Test MobileLiteVariant forward pass."""
        model = MobileLiteVariant(num_classes=313, base_channels=32)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (2, 313, 256, 256)
    
    def test_l2_regression_forward(self):
        """Test L2RegressionNet forward pass."""
        model = L2RegressionNet(base_channels=32)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (2, 2, 256, 256)
    
    def test_different_input_sizes(self):
        """Test models with different input sizes."""
        model = MobileLiteVariant(num_classes=313)
        model.eval()
        
        sizes = [128, 256, 512]
        
        for size in sizes:
            x = torch.randn(1, 1, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 313, size, size)
    
    def test_model_factory(self):
        """Test model factory function."""
        # Test paper model
        config = {'model_type': 'paper', 'num_classes': 313}
        model = get_model(config)
        assert isinstance(model, PaperNet)
        
        # Test mobile model
        config = {'model_type': 'mobile', 'num_classes': 313}
        model = get_model(config)
        assert isinstance(model, MobileLiteVariant)
        
        # Test L2 model
        config = {'model_type': 'l2'}
        model = get_model(config)
        assert isinstance(model, L2RegressionNet)
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = MobileLiteVariant(base_channels=16)
        count = count_parameters(model)
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_mobile_smaller_than_paper(self):
        """Test that mobile variant is smaller."""
        paper_model = PaperNet()
        mobile_model = MobileLiteVariant(base_channels=32)
        
        paper_params = count_parameters(paper_model)
        mobile_params = count_parameters(mobile_model)
        
        assert mobile_params < paper_params
    
    def test_batch_size_one(self):
        """Test models with batch size 1."""
        model = MobileLiteVariant()
        model.eval()
        
        x = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape[0] == 1
    
    def test_gradient_flow(self):
        """Test that gradients flow through models."""
        model = MobileLiteVariant(base_channels=16)
        model.train()
        
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestModelInitialization:
    """Test model initialization."""
    
    def test_weights_not_zero(self):
        """Test that initialized weights are not all zero."""
        model = MobileLiteVariant(base_channels=16)
        
        # At least some parameters should be non-zero
        # (bias can be zero, but weights should not be all zero)
        has_nonzero = False
        for param in model.parameters():
            if param.requires_grad and param.numel() > 0:
                if not torch.all(param == 0):
                    has_nonzero = True
                    break
        
        assert has_nonzero, "All parameters are zero"
    
    def test_reproducible_initialization(self):
        """Test that initialization is reproducible with same seed."""
        torch.manual_seed(42)
        model1 = MobileLiteVariant(base_channels=16)
        
        torch.manual_seed(42)
        model2 = MobileLiteVariant(base_channels=16)
        
        # Check first layer weights are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
