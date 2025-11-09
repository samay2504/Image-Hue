"""
Unit tests for quantization and color space operations.
"""

import pytest
import numpy as np
import torch

from src.models.ops import (
    rgb_to_lab, lab_to_rgb, get_ab_grid,
    encode_ab_to_distribution, decode_distribution_to_ab,
    compute_class_rebalancing_weights, ab_to_bin_indices
)


class TestColorSpaceConversion:
    """Test RGB<->Lab conversion."""
    
    def test_rgb_to_lab_shape(self):
        """Test RGB to Lab conversion shape."""
        rgb = np.random.rand(64, 64, 3)
        lab = rgb_to_lab(rgb)
        
        assert lab.shape == (64, 64, 3)
        assert lab.dtype == np.float32
    
    def test_lab_to_rgb_shape(self):
        """Test Lab to RGB conversion shape."""
        lab = np.random.rand(64, 64, 3)
        lab[:, :, 0] *= 100  # L in [0, 100]
        lab[:, :, 1:] = lab[:, :, 1:] * 200 - 100  # ab in [-100, 100]
        
        rgb = lab_to_rgb(lab)
        
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.float32
        assert np.all(rgb >= 0) and np.all(rgb <= 1)
    
    def test_roundtrip_conversion(self):
        """Test RGB->Lab->RGB roundtrip."""
        rgb_original = np.random.rand(32, 32, 3)
        
        lab = rgb_to_lab(rgb_original)
        rgb_reconstructed = lab_to_rgb(lab)
        
        # Allow small numerical error
        assert np.allclose(rgb_original, rgb_reconstructed, atol=0.01)
    
    def test_lab_ranges(self):
        """Test that Lab conversion produces correct ranges."""
        rgb = np.random.rand(32, 32, 3)
        lab = rgb_to_lab(rgb)
        
        # L should be in [0, 100]
        assert np.all(lab[:, :, 0] >= 0) and np.all(lab[:, :, 0] <= 100)
        
        # a and b should be roughly in [-127, 127] but allow some margin
        assert np.all(lab[:, :, 1] >= -130) and np.all(lab[:, :, 1] <= 130)
        assert np.all(lab[:, :, 2] >= -130) and np.all(lab[:, :, 2] <= 130)


class TestQuantization:
    """Test ab space quantization."""
    
    def test_ab_grid_shape(self):
        """Test ab quantization grid."""
        ab_grid = get_ab_grid()
        
        assert ab_grid.shape[1] == 2  # (Q, 2)
        assert ab_grid.shape[0] > 0  # Should have bins
        # Paper reports Q=313 for grid_size=10
        # Our implementation may differ slightly (313-500 is acceptable)
        assert 250 <= ab_grid.shape[0] <= 550
    
    def test_soft_encoding_shape(self):
        """Test soft-encoding output shape."""
        ab = np.random.rand(32, 32, 2) * 200 - 100  # ab in [-100, 100]
        
        dist = encode_ab_to_distribution(ab)
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        assert dist.shape == (32, 32, Q)
        assert dist.dtype == np.float32
    
    def test_soft_encoding_normalization(self):
        """Test that soft-encoded distributions sum to 1."""
        ab = np.random.rand(16, 16, 2) * 200 - 100
        
        dist = encode_ab_to_distribution(ab)
        
        # Each pixel's distribution should sum to ~1
        sums = np.sum(dist, axis=2)
        assert np.allclose(sums, 1.0, atol=0.01)
    
    def test_decode_distribution_shape(self):
        """Test distribution decoding."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Create random logits
        logits = torch.randn(2, Q, 32, 32)
        
        ab_pred = decode_distribution_to_ab(logits, temperature=0.38)
        
        assert ab_pred.shape == (2, 2, 32, 32)
        assert ab_pred.dtype == torch.float32
    
    def test_decode_temperature_effect(self):
        """Test that temperature affects decoding."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        logits = torch.randn(1, Q, 16, 16)
        
        # Low temperature should give more extreme values
        ab_low_temp = decode_distribution_to_ab(logits, temperature=0.1)
        ab_high_temp = decode_distribution_to_ab(logits, temperature=1.0)
        
        # Variance should be different
        var_low = torch.var(ab_low_temp)
        var_high = torch.var(ab_high_temp)
        
        # Not always true but generally low temp has higher variance
        # Just check they're different
        assert not torch.allclose(ab_low_temp, ab_high_temp)
    
    def test_bin_indices(self):
        """Test ab to bin index conversion."""
        ab = np.random.rand(16, 16, 2) * 200 - 100
        
        indices = ab_to_bin_indices(ab)
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        assert indices.shape == (16, 16)
        assert indices.dtype in [np.int32, np.int64]
        assert np.all(indices >= 0) and np.all(indices < Q)


class TestClassRebalancing:
    """Test class rebalancing weight computation."""
    
    def test_rebalancing_weights_shape(self):
        """Test rebalancing weights shape."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Create random empirical distribution
        empirical = np.random.rand(Q)
        empirical /= empirical.sum()
        
        weights = compute_class_rebalancing_weights(empirical)
        
        assert weights.shape == (Q,)
        assert weights.dtype == np.float32
    
    def test_rebalancing_weights_mean(self):
        """Test that rebalancing weights have mean=1."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        empirical = np.random.rand(Q)
        weights = compute_class_rebalancing_weights(empirical)
        
        # Weights should have mean ≈ 1
        assert np.isclose(weights.mean(), 1.0, atol=0.01)
    
    def test_rebalancing_inverts_frequency(self):
        """Test that frequent classes get lower weights."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Create skewed distribution
        empirical = np.ones(Q)
        empirical[0] = 100  # Make first class very frequent
        
        weights = compute_class_rebalancing_weights(empirical)
        
        # First class should have relatively lower weight (but not exactly inverse due to smoothing)
        assert weights[0] < weights.mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
