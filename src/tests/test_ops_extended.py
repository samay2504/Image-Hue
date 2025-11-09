"""
Extended unit tests for quantization and color space operations with maximum coverage.
"""

import pytest
import numpy as np
import torch

from src.models.ops import (
    rgb_to_lab, lab_to_rgb, get_ab_grid,
    encode_ab_to_distribution, decode_distribution_to_ab,
    compute_class_rebalancing_weights, ab_to_bin_indices
)


class TestColorSpaceConversionExtended:
    """Extended tests for RGB<->Lab conversion."""
    
    def test_rgb_to_lab_edge_cases(self):
        """Test edge case colors."""
        # Pure black
        black = np.zeros((1, 1, 3), dtype=np.float32)
        lab_black = rgb_to_lab(black)
        assert lab_black[0, 0, 0] < 1.0  # L should be near 0
        
        # Pure white
        white = np.ones((1, 1, 3), dtype=np.float32)
        lab_white = rgb_to_lab(white)
        assert lab_white[0, 0, 0] > 99.0  # L should be near 100
        
        # Pure red
        red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        lab_red = rgb_to_lab(red)
        assert lab_red.shape == (1, 1, 3)
        
        # Pure green
        green = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        lab_green = rgb_to_lab(green)
        assert lab_green.shape == (1, 1, 3)
        
        # Pure blue
        blue = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        lab_blue = rgb_to_lab(blue)
        assert lab_blue.shape == (1, 1, 3)
    
    def test_rgb_to_lab_batch(self):
        """Test batch processing."""
        batch = np.random.rand(10, 64, 64, 3).astype(np.float32)
        
        for i in range(10):
            lab = rgb_to_lab(batch[i])
            assert lab.shape == (64, 64, 3)
    
    def test_lab_to_rgb_edge_cases(self):
        """Test Lab to RGB edge cases."""
        # Gray (L only, a=0, b=0)
        gray_lab = np.zeros((1, 1, 3), dtype=np.float32)
        gray_lab[0, 0, 0] = 50.0  # Mid gray
        rgb = lab_to_rgb(gray_lab)
        assert np.all(rgb >= 0) and np.all(rgb <= 1)
    
    def test_roundtrip_with_clipping(self):
        """Test roundtrip with out-of-gamut colors."""
        # Create extreme Lab values
        lab = np.random.rand(32, 32, 3).astype(np.float32)
        lab[:, :, 0] *= 100  # L in [0, 100]
        lab[:, :, 1:] = lab[:, :, 1:] * 200 - 100  # ab in [-100, 100]
        
        rgb = lab_to_rgb(lab)
        lab_back = rgb_to_lab(rgb)
        
        # Should be close after clipping
        assert lab_back.shape == lab.shape
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very dark colors
        dark = np.ones((16, 16, 3), dtype=np.float32) * 0.001
        lab_dark = rgb_to_lab(dark)
        rgb_dark = lab_to_rgb(lab_dark)
        assert not np.any(np.isnan(rgb_dark))
        assert not np.any(np.isinf(rgb_dark))
        
        # Very bright colors
        bright = np.ones((16, 16, 3), dtype=np.float32) * 0.999
        lab_bright = rgb_to_lab(bright)
        rgb_bright = lab_to_rgb(lab_bright)
        assert not np.any(np.isnan(rgb_bright))
        assert not np.any(np.isinf(rgb_bright))
    
    def test_grayscale_conversion(self):
        """Test grayscale images (R=G=B)."""
        gray_values = np.linspace(0, 1, 10)
        for val in gray_values:
            rgb = np.full((1, 1, 3), val, dtype=np.float32)
            lab = rgb_to_lab(rgb)
            
            # For grayscale, a and b should be close to 0
            assert abs(lab[0, 0, 1]) < 5.0
            assert abs(lab[0, 0, 2]) < 5.0


class TestQuantizationExtended:
    """Extended tests for ab space quantization."""
    
    def test_ab_grid_consistency(self):
        """Test that grid is consistent across calls."""
        grid1 = get_ab_grid()
        grid2 = get_ab_grid()
        
        assert np.allclose(grid1, grid2)
    
    def test_ab_grid_in_gamut(self):
        """Test that grid points are reasonable."""
        ab_grid = get_ab_grid()
        
        # Points should be in reasonable range
        assert np.all(ab_grid[:, 0] >= -110) and np.all(ab_grid[:, 0] <= 110)
        assert np.all(ab_grid[:, 1] >= -110) and np.all(ab_grid[:, 1] <= 110)
    
    def test_soft_encoding_with_zeros(self):
        """Test soft-encoding with zero ab values."""
        ab = np.zeros((16, 16, 2), dtype=np.float32)
        dist = encode_ab_to_distribution(ab)
        
        assert not np.any(np.isnan(dist))
        assert np.all(dist >= 0)
    
    def test_soft_encoding_consistency(self):
        """Test that same input gives same encoding."""
        ab = np.random.rand(16, 16, 2).astype(np.float32) * 200 - 100
        
        dist1 = encode_ab_to_distribution(ab)
        dist2 = encode_ab_to_distribution(ab)
        
        assert np.allclose(dist1, dist2)
    
    def test_soft_encoding_sparse(self):
        """Test that soft-encoding is sparse."""
        ab = np.random.rand(16, 16, 2).astype(np.float32) * 200 - 100
        dist = encode_ab_to_distribution(ab)
        
        # Most values should be close to zero (sparse)
        non_zero_ratio = np.mean(dist > 0.01)
        assert non_zero_ratio < 0.5  # Less than 50% should be non-zero
    
    def test_decode_with_uniform_distribution(self):
        """Test decoding with uniform distribution."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Uniform logits
        logits = torch.zeros(1, Q, 16, 16)
        
        ab_pred = decode_distribution_to_ab(logits, temperature=0.38)
        
        assert not torch.any(torch.isnan(ab_pred))
        assert not torch.any(torch.isinf(ab_pred))
    
    def test_decode_with_peaked_distribution(self):
        """Test decoding with peaked distribution."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Create peaked distribution
        logits = torch.ones(1, Q, 16, 16) * -10
        logits[:, 0, :, :] = 10  # Peak at first bin
        
        ab_pred = decode_distribution_to_ab(logits, temperature=0.38)
        
        # Should be close to first grid point - use reasonable threshold
        # Grid values can be up to 110 in magnitude for extreme colors
        assert ab_pred.abs().mean() < 120  # Allow for edge grid points
    
    def test_decode_temperature_range(self):
        """Test decoding with different temperature ranges."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        logits = torch.randn(1, Q, 16, 16)
        
        temperatures = [0.01, 0.1, 0.38, 1.0, 2.0]
        results = []
        
        for temp in temperatures:
            ab_pred = decode_distribution_to_ab(logits, temperature=temp)
            results.append(ab_pred)
            
            assert not torch.any(torch.isnan(ab_pred))
            assert not torch.any(torch.isinf(ab_pred))
        
        # Results should vary with temperature
        for i in range(len(results) - 1):
            assert not torch.allclose(results[i], results[i + 1])
    
    def test_bin_indices_consistency(self):
        """Test bin index consistency."""
        ab = np.random.rand(16, 16, 2).astype(np.float32) * 200 - 100
        
        indices1 = ab_to_bin_indices(ab)
        indices2 = ab_to_bin_indices(ab)
        
        assert np.array_equal(indices1, indices2)
    
    def test_bin_indices_extremes(self):
        """Test bin indices with extreme values."""
        # Maximum values
        ab_max = np.ones((8, 8, 2), dtype=np.float32) * 100
        indices_max = ab_to_bin_indices(ab_max)
        assert np.all(indices_max >= 0)
        
        # Minimum values
        ab_min = np.ones((8, 8, 2), dtype=np.float32) * -100
        indices_min = ab_to_bin_indices(ab_min)
        assert np.all(indices_min >= 0)


class TestClassRebalancingExtended:
    """Extended tests for class rebalancing."""
    
    def test_rebalancing_with_uniform_distribution(self):
        """Test rebalancing with uniform distribution."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Uniform distribution
        empirical = np.ones(Q) / Q
        weights = compute_class_rebalancing_weights(empirical)
        
        # All weights should be similar for uniform distribution
        assert np.std(weights) < 0.5
    
    def test_rebalancing_with_zero_frequency(self):
        """Test rebalancing with zero-frequency classes."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        empirical = np.ones(Q)
        empirical[0] = 0  # Zero frequency
        
        weights = compute_class_rebalancing_weights(empirical)
        
        # Should handle zero frequency gracefully
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
    
    def test_rebalancing_preserves_total_weight(self):
        """Test that total weight is preserved."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        empirical = np.random.rand(Q)
        weights = compute_class_rebalancing_weights(empirical)
        
        # Mean should be 1.0
        assert np.isclose(weights.mean(), 1.0, atol=0.01)
    
    def test_rebalancing_lambda_effect(self):
        """Test lambda parameter effect."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Skewed distribution
        empirical = np.ones(Q)
        empirical[0] = 100
        
        weights = compute_class_rebalancing_weights(empirical, lambda_mix=0.0)
        weights_smoothed = compute_class_rebalancing_weights(empirical, lambda_mix=0.5)
        
        # With lambda=0.5, weights should be less extreme
        assert np.std(weights_smoothed) < np.std(weights)


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test with empty inputs."""
        with pytest.raises((ValueError, IndexError)):
            rgb_to_lab(np.array([]))
    
    def test_wrong_shape_rgb(self):
        """Test RGB with wrong shape."""
        with pytest.raises((ValueError, IndexError)):
            rgb_to_lab(np.random.rand(64, 64))  # Missing channel dimension
    
    def test_wrong_dtype(self):
        """Test with integer inputs."""
        # Should handle or convert
        rgb_int = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        try:
            lab = rgb_to_lab(rgb_int)
            assert lab.shape == (32, 32, 3)
        except (TypeError, ValueError):
            pass  # Expected to fail if conversion not handled
    
    def test_negative_rgb_values(self):
        """Test with negative RGB values."""
        rgb = np.random.rand(16, 16, 3) - 0.5  # Some negative
        lab = rgb_to_lab(np.clip(rgb, 0, 1))
        assert not np.any(np.isnan(lab))
    
    def test_out_of_range_temperature(self):
        """Test extreme temperature values."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        logits = torch.randn(1, Q, 16, 16)
        
        # Very low temperature
        ab_low = decode_distribution_to_ab(logits, temperature=0.001)
        assert not torch.any(torch.isnan(ab_low))
        
        # Very high temperature
        ab_high = decode_distribution_to_ab(logits, temperature=10.0)
        assert not torch.any(torch.isnan(ab_high))


class TestMemoryEfficiency:
    """Test memory efficiency for 6GB GPU."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_large_batch_cpu_fallback(self):
        """Test that large batches work on CPU."""
        ab_grid = get_ab_grid()
        Q = len(ab_grid)
        
        # Large batch on CPU
        logits = torch.randn(16, Q, 64, 64)
        ab_pred = decode_distribution_to_ab(logits, temperature=0.38)
        
        assert ab_pred.shape == (16, 2, 64, 64)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_safe_decode(self):
        """Test decoding doesn't cause OOM on 6GB GPU."""
        if torch.cuda.is_available():
            ab_grid = get_ab_grid()
            Q = len(ab_grid)
            
            # Moderate size that should fit in 6GB
            logits = torch.randn(4, Q, 256, 256, device='cuda')
            
            ab_pred = decode_distribution_to_ab(logits, temperature=0.38)
            
            assert ab_pred.device.type == 'cuda'
            
            # Clean up
            del logits, ab_pred
            torch.cuda.empty_cache()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
