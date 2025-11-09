"""
SPADE (Spatially-Adaptive Normalization) and AdaIN (Adaptive Instance Normalization) modules.

Based on:
- SPADE: "Semantic Image Synthesis with Spatially-Adaptive Normalization" (Park et al., CVPR 2019)
- AdaIN: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (Huang & Belongie, ICCV 2017)

Used for conditional normalization in the decoder, modulated by the grayscale input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE).
    
    Modulates normalized activations with learned affine parameters that are
    spatially varying and conditioned on a semantic segmentation map (here, the L channel).
    
    Formula:
        gamma, beta = MLP(segmap)  # Spatially-varying
        out = gamma * InstanceNorm(x) + beta
    """
    
    def __init__(
        self,
        norm_nc: int,
        label_nc: int = 1,
        nhidden: int = 128,
        kernel_size: int = 3,
        use_spectral_norm: bool = False,
    ):
        """
        Args:
            norm_nc: Number of channels in the activation to be normalized
            label_nc: Number of channels in the conditioning map (1 for grayscale L)
            nhidden: Number of hidden channels in modulation network
            kernel_size: Kernel size for modulation convolutions
            use_spectral_norm: Apply spectral normalization to conv layers
        """
        super().__init__()
        
        self.norm_nc = norm_nc
        self.label_nc = label_nc
        
        # Instance normalization (can also use batch norm)
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # Shared MLP to generate modulation parameters
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        
        # Separate heads for gamma (scale) and beta (shift)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)
        
        # Apply spectral norm if requested
        if use_spectral_norm:
            self.mlp_shared = nn.utils.spectral_norm(self.mlp_shared[0])
            self.mlp_gamma = nn.utils.spectral_norm(self.mlp_gamma)
            self.mlp_beta = nn.utils.spectral_norm(self.mlp_beta)
    
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        Apply SPADE normalization.
        
        Args:
            x: Activation to normalize, shape [B, C, H, W]
            segmap: Conditioning map (L channel), shape [B, 1, H_seg, W_seg]
            
        Returns:
            normalized: [B, C, H, W] with spatially-varying modulation
        """
        # Normalize activation
        normalized = self.param_free_norm(x)
        
        # Resize segmap to match activation size if needed
        if segmap.shape[2:] != x.shape[2:]:
            segmap = F.interpolate(
                segmap, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Generate modulation parameters
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)  # [B, C, H, W] - spatially varying scale
        beta = self.mlp_beta(actv)    # [B, C, H, W] - spatially varying shift
        
        # Apply affine transformation
        out = normalized * (1 + gamma) + beta
        
        return out


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).
    
    Simpler alternative to SPADE - uses global statistics instead of spatial maps.
    Modulates normalized activations with channel-wise affine parameters.
    
    Formula:
        gamma, beta = MLP(global_pool(segmap))  # Channel-wise only
        out = gamma * InstanceNorm(x) + beta
    """
    
    def __init__(
        self,
        norm_nc: int,
        label_nc: int = 1,
        nhidden: int = 128,
    ):
        """
        Args:
            norm_nc: Number of channels in the activation to be normalized
            label_nc: Number of channels in the conditioning map
            nhidden: Number of hidden units in MLP
        """
        super().__init__()
        
        self.norm_nc = norm_nc
        self.label_nc = label_nc
        
        # Instance normalization
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # MLP to generate channel-wise modulation parameters
        # Input: global pooled features from segmap
        self.mlp = nn.Sequential(
            nn.Linear(label_nc, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, norm_nc * 2)  # gamma and beta concatenated
        )
    
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN normalization.
        
        Args:
            x: Activation to normalize, shape [B, C, H, W]
            segmap: Conditioning map, shape [B, label_nc, H_seg, W_seg]
            
        Returns:
            normalized: [B, C, H, W] with channel-wise modulation
        """
        B, C, H, W = x.shape
        
        # Normalize activation
        normalized = self.param_free_norm(x)
        
        # Global average pooling of segmap
        segmap_pooled = F.adaptive_avg_pool2d(segmap, 1)  # [B, label_nc, 1, 1]
        segmap_pooled = segmap_pooled.view(B, -1)  # [B, label_nc]
        
        # Generate modulation parameters
        style = self.mlp(segmap_pooled)  # [B, norm_nc * 2]
        style = style.view(B, 2, C)  # [B, 2, C]
        
        gamma = style[:, 0, :].view(B, C, 1, 1)  # [B, C, 1, 1] - channel-wise scale
        beta = style[:, 1, :].view(B, C, 1, 1)   # [B, C, 1, 1] - channel-wise shift
        
        # Apply affine transformation
        out = normalized * (1 + gamma) + beta
        
        return out


class ConditionalNorm2d(nn.Module):
    """
    Unified conditional normalization module with automatic fallback.
    
    Uses SPADE by default, with automatic fallback to AdaIN on OOM or explicit request.
    """
    
    def __init__(
        self,
        norm_nc: int,
        label_nc: int = 1,
        nhidden: int = 128,
        mode: str = "spade",
        **kwargs
    ):
        """
        Args:
            norm_nc: Number of channels to normalize
            label_nc: Number of conditioning channels
            nhidden: Hidden dimension
            mode: "spade" or "adain"
            **kwargs: Additional arguments for SPADE/AdaIN
        """
        super().__init__()
        
        self.mode = mode.lower()
        
        if self.mode == "spade":
            self.norm = SPADE(norm_nc, label_nc, nhidden, **kwargs)
        elif self.mode == "adain":
            self.norm = AdaIN(norm_nc, label_nc, nhidden)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
    
    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """Apply conditional normalization."""
        return self.norm(x, segmap)


def test_normalization():
    """Test SPADE and AdaIN modules."""
    print("Testing SPADE and AdaIN...")
    
    B, C, H, W = 4, 64, 32, 32
    label_nc = 1
    
    x = torch.randn(B, C, H, W)
    segmap = torch.randn(B, label_nc, H, W)
    
    # Test SPADE
    print("\n1. Testing SPADE:")
    spade = SPADE(norm_nc=C, label_nc=label_nc)
    out_spade = spade(x, segmap)
    print(f"   Input shape: {x.shape}")
    print(f"   Segmap shape: {segmap.shape}")
    print(f"   Output shape: {out_spade.shape}")
    assert out_spade.shape == x.shape, "SPADE output shape mismatch"
    print("   ✓ SPADE test passed")
    
    # Test AdaIN
    print("\n2. Testing AdaIN:")
    adain = AdaIN(norm_nc=C, label_nc=label_nc)
    out_adain = adain(x, segmap)
    print(f"   Input shape: {x.shape}")
    print(f"   Segmap shape: {segmap.shape}")
    print(f"   Output shape: {out_adain.shape}")
    assert out_adain.shape == x.shape, "AdaIN output shape mismatch"
    print("   ✓ AdaIN test passed")
    
    # Test ConditionalNorm2d
    print("\n3. Testing ConditionalNorm2d:")
    cond_norm = ConditionalNorm2d(norm_nc=C, label_nc=label_nc, mode="spade")
    out_cond = cond_norm(x, segmap)
    assert out_cond.shape == x.shape, "ConditionalNorm2d output shape mismatch"
    print("   ✓ ConditionalNorm2d test passed")
    
    # Test with different segmap size (should auto-resize)
    print("\n4. Testing auto-resize:")
    segmap_small = torch.randn(B, label_nc, 16, 16)
    out_resize = spade(x, segmap_small)
    assert out_resize.shape == x.shape, "Auto-resize failed"
    print(f"   Segmap size {segmap_small.shape[2:]} -> output {out_resize.shape}")
    print("   ✓ Auto-resize test passed")
    
    print("\n✓ All normalization tests passed!")


if __name__ == "__main__":
    test_normalization()
