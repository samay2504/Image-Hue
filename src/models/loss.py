"""
Perceptual and combined losses for modern colorization.

Implements VGG-based perceptual loss and combined loss functions.
"""

import logging
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    
    Computes L1 distance between multi-layer VGG features of predicted
    and ground truth images. Encourages perceptual similarity.
    
    Reference: Johnson et al., "Perceptual Losses for Real-Time Style 
    Transfer and Super-Resolution", ECCV 2016.
    """
    
    def __init__(
        self,
        layers: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        normalize_input: bool = True,
        require_grad: bool = False,
    ):
        """
        Args:
            layers: VGG layer names to extract features from.
                    Default: ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
            weights: Per-layer loss weights. Default: equal weights.
            normalize_input: Normalize input with ImageNet stats.
            require_grad: Allow gradients through VGG (for fine-tuning).
        """
        super().__init__()
        
        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        
        if weights is None:
            weights = [1.0] * len(layers)
        
        assert len(layers) == len(weights), "Layers and weights must match"
        
        self.layers = layers
        self.weights = weights
        self.normalize_input = normalize_input
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Build feature extractor with named layers
        self.feature_extractor = nn.ModuleDict()
        
        layer_mapping = {
            "relu1_1": 1, "relu1_2": 3,
            "relu2_1": 6, "relu2_2": 8,
            "relu3_1": 11, "relu3_2": 13, "relu3_3": 15,
            "relu4_1": 18, "relu4_2": 20, "relu4_3": 22,
            "relu5_1": 25, "relu5_2": 27, "relu5_3": 29,
        }
        
        max_layer_idx = max(layer_mapping[layer] for layer in layers)
        
        for layer_name in layers:
            idx = layer_mapping[layer_name]
            self.feature_extractor[layer_name] = nn.Sequential(
                *list(vgg.children())[:idx+1]
            )
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = require_grad
        
        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        logger.info(f"VGGPerceptualLoss initialized:")
        logger.info(f"  Layers: {layers}")
        logger.info(f"  Weights: {weights}")
        logger.info(f"  Normalize: {normalize_input}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with ImageNet stats."""
        return (x - self.mean) / self.std
    
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers."""
        features = {}
        for layer_name, extractor in self.feature_extractor.items():
            features[layer_name] = extractor(x)
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: [B, 3, H, W] predicted RGB image in [0, 1]
            target: [B, 3, H, W] target RGB image in [0, 1]
            
        Returns:
            Scalar perceptual loss.
        """
        # Normalize inputs
        if self.normalize_input:
            pred = self._normalize(pred)
            target = self._normalize(target)
        
        # Extract features
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        # Compute weighted L1 loss per layer
        loss = 0.0
        for layer_name, weight in zip(self.layers, self.weights):
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            
            # L1 distance
            layer_loss = F.l1_loss(pred_feat, target_feat)
            loss += weight * layer_loss
        
        return loss


class CombinedColorizationLoss(nn.Module):
    """
    Combined loss for colorization training.
    
    Combines classification/regression loss with perceptual loss.
    """
    
    def __init__(
        self,
        mode: str = "classification",
        use_perceptual: bool = True,
        perceptual_weight: float = 0.1,
        perceptual_layers: Optional[List[str]] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            mode: "classification" or "regression"
            use_perceptual: Include perceptual loss
            perceptual_weight: Weight for perceptual loss
            perceptual_layers: VGG layers for perceptual loss
            class_weights: [Q] class weights for classification (optional)
        """
        super().__init__()
        
        self.mode = mode
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        
        if mode == "classification":
            # Cross-entropy loss with optional class weights
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction="mean"
            )
        else:
            # L1 loss for regression
            self.l1_loss = nn.L1Loss()
        
        # Perceptual loss
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss(
                layers=perceptual_layers,
                normalize_input=True,
            )
        
        logger.info(f"CombinedColorizationLoss initialized:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Perceptual: {use_perceptual} (weight={perceptual_weight})")
    
    def _lab_to_rgb(
        self,
        L: torch.Tensor,
        ab: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert Lab to RGB for perceptual loss.
        
        Simplified conversion assuming L in [-1,1], ab in [-1,1] scaled to Lab ranges.
        For accurate conversion, use proper color space ops from ops.py.
        """
        # Scale to approximate Lab ranges
        L_scaled = (L + 1) * 50  # [0, 100]
        a_scaled = ab[:, 0:1] * 110  # [-110, 110]
        b_scaled = ab[:, 1:2] * 110  # [-110, 110]
        
        # Simplified Lab->RGB (approximation for loss computation)
        # In production, use proper conversion from ops.py
        y = (L_scaled + 16) / 116
        x = a_scaled / 500 + y
        z = y - b_scaled / 200
        
        # XYZ to RGB (simplified D65)
        X = x ** 3
        Y = y ** 3
        Z = z ** 3
        
        R = X * 3.2406 + Y * (-1.5372) + Z * (-0.4986)
        G = X * (-0.9689) + Y * 1.8758 + Z * 0.0415
        B = X * 0.0557 + Y * (-0.2040) + Z * 1.0570
        
        RGB = torch.cat([R, G, B], dim=1)
        RGB = torch.clamp(RGB, 0, 1)
        
        return RGB
    
    def forward(
        self,
        pred_logits_or_ab: torch.Tensor,
        target_ab: torch.Tensor,
        L: torch.Tensor,
        target_bins: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_logits_or_ab: [B, Q, H, W] logits or [B, 2, H, W] ab
            target_ab: [B, 2, H, W] ground truth ab channels
            L: [B, 1, H, W] L channel (for perceptual loss)
            target_bins: [B, H, W] ground truth bin indices (for classification)
            
        Returns:
            Dict with keys:
                - "loss": Total loss
                - "classification_loss" or "regression_loss": Main loss
                - "perceptual_loss": Perceptual loss (if enabled)
        """
        losses = {}
        
        # Main loss (classification or regression)
        if self.mode == "classification":
            assert target_bins is not None, "Classification requires target_bins"
            
            # Cross-entropy loss
            # pred_logits: [B, Q, H, W], target_bins: [B, H, W]
            ce_loss = self.ce_loss(pred_logits_or_ab, target_bins)
            losses["classification_loss"] = ce_loss
            main_loss = ce_loss
            
            # For perceptual loss, need to convert logits to ab
            # Use decode_distribution_to_ab (annealed_mean)
            from src.models.ops import decode_distribution_to_ab
            pred_ab = decode_distribution_to_ab(pred_logits_or_ab, temperature=0.38)
        
        else:
            # L1 regression loss
            l1_loss = self.l1_loss(pred_logits_or_ab, target_ab)
            losses["regression_loss"] = l1_loss
            main_loss = l1_loss
            pred_ab = pred_logits_or_ab
        
        # Perceptual loss
        if self.use_perceptual:
            # Convert Lab to RGB
            pred_rgb = self._lab_to_rgb(L, pred_ab)
            target_rgb = self._lab_to_rgb(L, target_ab)
            
            perceptual_loss = self.perceptual_loss(pred_rgb, target_rgb)
            losses["perceptual_loss"] = perceptual_loss
            
            total_loss = main_loss + self.perceptual_weight * perceptual_loss
        else:
            total_loss = main_loss
        
        losses["loss"] = total_loss
        
        return losses


def test_loss_functions():
    """Test loss functions."""
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    try:
        # Test VGG perceptual loss
        print("\n1. Testing VGGPerceptualLoss:")
        perceptual_loss = VGGPerceptualLoss(
            layers=["relu2_2", "relu3_3"],
            weights=[1.0, 1.0],
        )
        
        B, H, W = 2, 224, 224
        pred_rgb = torch.rand(B, 3, H, W)
        target_rgb = torch.rand(B, 3, H, W)
        
        loss_val = perceptual_loss(pred_rgb, target_rgb)
        print(f"   Input shape: {pred_rgb.shape}")
        print(f"   Loss value: {loss_val.item():.6f}")
        assert loss_val.item() >= 0, "Loss should be non-negative"
        print("   ✓ VGGPerceptualLoss test passed")
        
        # Test combined loss - classification mode
        print("\n2. Testing CombinedColorizationLoss (classification):")
        combined_loss_cls = CombinedColorizationLoss(
            mode="classification",
            use_perceptual=True,
            perceptual_weight=0.1,
        )
        
        Q = 313
        pred_logits = torch.randn(B, Q, H, W)
        target_ab = torch.randn(B, 2, H, W)
        L = torch.randn(B, 1, H, W)
        target_bins = torch.randint(0, Q, (B, H, W))
        
        losses = combined_loss_cls(pred_logits, target_ab, L, target_bins)
        
        print(f"   Total loss: {losses['loss'].item():.6f}")
        print(f"   Classification loss: {losses['classification_loss'].item():.6f}")
        print(f"   Perceptual loss: {losses['perceptual_loss'].item():.6f}")
        assert all(v.item() >= 0 for v in losses.values()), "All losses should be non-negative"
        print("   ✓ Classification mode test passed")
        
        # Test combined loss - regression mode
        print("\n3. Testing CombinedColorizationLoss (regression):")
        combined_loss_reg = CombinedColorizationLoss(
            mode="regression",
            use_perceptual=True,
            perceptual_weight=0.05,
        )
        
        pred_ab = torch.randn(B, 2, H, W)
        
        losses_reg = combined_loss_reg(pred_ab, target_ab, L, target_bins=None)
        
        print(f"   Total loss: {losses_reg['loss'].item():.6f}")
        print(f"   Regression loss: {losses_reg['regression_loss'].item():.6f}")
        print(f"   Perceptual loss: {losses_reg['perceptual_loss'].item():.6f}")
        assert all(v.item() >= 0 for v in losses_reg.values()), "All losses should be non-negative"
        print("   ✓ Regression mode test passed")
        
        # Test without perceptual loss
        print("\n4. Testing without perceptual loss:")
        combined_loss_no_perceptual = CombinedColorizationLoss(
            mode="regression",
            use_perceptual=False,
        )
        
        losses_no_perceptual = combined_loss_no_perceptual(
            pred_ab, target_ab, L, target_bins=None
        )
        
        assert "perceptual_loss" not in losses_no_perceptual
        print(f"   Total loss: {losses_no_perceptual['loss'].item():.6f}")
        print("   ✓ No perceptual loss test passed")
        
        print("\n" + "=" * 60)
        print("✓ All loss function tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loss_functions()
