"""
Unit tests for modern transformer-based colorization architecture.

Tests encoder, decoder, normalization, loss, and end-to-end pipeline.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os

from src.models.encoder_transformer import TransformerEncoder
from src.models.decoder_spade import SPADEDecoder
from src.models.normalization import SPADE, AdaIN, ConditionalNorm2d
from src.models.modern_colorizer import ModernColorizer
from src.models.loss import VGGPerceptualLoss, CombinedColorizationLoss
from src.utils.hf_model_cache import HFModelLoader


# Skip tests requiring HuggingFace models if offline or not available
def check_hf_available():
    """Check if HuggingFace models are available (cached or downloadable)."""
    try:
        loader = HFModelLoader()
        # Try to load a tiny model with local_only
        loader.load_model("tiny", local_only=True)
        return True
    except:
        return False


requires_hf = pytest.mark.skipif(
    not check_hf_available(),
    reason="HuggingFace models not available (offline or not cached)"
)


class TestTransformerEncoder:
    """Test transformer encoder."""

    @requires_hf
    def test_encoder_initialization(self):
        """Test encoder can be initialized."""
        encoder = TransformerEncoder(
            size="tiny",
            pretrained=True,  # Use pretrained for tests
            local_only=True,
        )
        assert encoder is not None
        assert isinstance(encoder, nn.Module)

    @requires_hf
    def test_encoder_forward_shape(self):
        """Test encoder output shapes."""
        encoder = TransformerEncoder(
            size="tiny",
            pretrained=True,  # Use pretrained
            local_only=True,
        )

        B, H, W = 2, 224, 224
        L = torch.randn(B, 1, H, W)

        features = encoder(L)

        assert isinstance(features, list)
        assert len(features) == 4  # Multi-scale features

        # Check feature shapes have correct batch and spatial dims
        for i, feat in enumerate(features):
            assert feat.shape[0] == B
            assert len(feat.shape) == 4  # [B, C, H, W]
            assert feat.shape[2] > 0 and feat.shape[3] > 0

    @requires_hf
    def test_encoder_freeze_unfreeze(self):
        """Test freeze/unfreeze functionality."""
        encoder = TransformerEncoder(
            size="tiny",
            pretrained=True,
            local_only=True,
        )

        # Count initial trainable params
        initial_trainable = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )

        # Freeze prefix blocks
        encoder.freeze_prefix_blocks(6)
        frozen_trainable = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        assert frozen_trainable < initial_trainable

        # Unfreeze
        encoder.unfreeze_all()
        unfrozen_trainable = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        assert unfrozen_trainable == initial_trainable


class TestNormalization:
    """Test SPADE and AdaIN normalization."""

    def test_spade_forward(self):
        """Test SPADE normalization."""
        B, C, H, W = 2, 64, 32, 32

        spade = SPADE(
            norm_nc=C,
            label_nc=1,
            nhidden=128,
        )

        x = torch.randn(B, C, H, W)
        segmap = torch.randn(B, 1, H, W)

        output = spade(x, segmap)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_adain_forward(self):
        """Test AdaIN normalization."""
        B, C, H, W = 2, 64, 32, 32

        adain = AdaIN(
            norm_nc=C,
            label_nc=1,
            nhidden=128,
        )

        x = torch.randn(B, C, H, W)
        segmap = torch.randn(B, 1, H, W)

        output = adain(x, segmap)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_conditional_norm_modes(self):
        """Test ConditionalNorm2d with both modes."""
        B, C, H, W = 2, 64, 32, 32

        # SPADE mode
        norm_spade = ConditionalNorm2d(
            norm_nc=C,
            label_nc=1,
            mode="spade",
        )

        x = torch.randn(B, C, H, W)
        segmap = torch.randn(B, 1, H, W)

        output_spade = norm_spade(x, segmap)
        assert output_spade.shape == x.shape

        # AdaIN mode
        norm_adain = ConditionalNorm2d(
            norm_nc=C,
            label_nc=1,
            mode="adain",
        )

        output_adain = norm_adain(x, segmap)
        assert output_adain.shape == x.shape


class TestSPADEDecoder:
    """Test SPADE decoder."""

    def test_decoder_initialization(self):
        """Test decoder can be initialized."""
        encoder_channels = [192, 384, 512, 768]

        decoder = SPADEDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[512, 256, 128, 64],
            num_output_channels=313,
            label_nc=1,
        )

        assert decoder is not None
        assert isinstance(decoder, nn.Module)

    def test_decoder_forward_shape(self):
        """Test decoder output shape."""
        B, H, W = 2, 224, 224
        encoder_channels = [192, 384, 512, 768]

        # Create dummy encoder features
        encoder_features = [
            torch.randn(B, 192, H // 16, W // 16),
            torch.randn(B, 384, H // 8, W // 8),
            torch.randn(B, 512, H // 4, W // 4),
            torch.randn(B, 768, H // 2, W // 2),
        ]

        L_cond = torch.randn(B, 1, H, W)

        decoder = SPADEDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[512, 256, 128, 64],
            num_output_channels=313,
            label_nc=1,
        )

        output = decoder(encoder_features, L_cond)

        assert output.shape == (B, 313, H, W)
        assert not torch.isnan(output).any()

    def test_decoder_different_output_channels(self):
        """Test decoder with regression output (2 channels)."""
        B, H, W = 2, 224, 224
        encoder_channels = [192, 384, 512, 768]

        encoder_features = [
            torch.randn(B, 192, H // 16, W // 16),
            torch.randn(B, 384, H // 8, W // 8),
            torch.randn(B, 512, H // 4, W // 4),
            torch.randn(B, 768, H // 2, W // 2),
        ]

        L_cond = torch.randn(B, 1, H, W)

        decoder = SPADEDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[512, 256, 128, 64],
            num_output_channels=2,  # Regression mode
            label_nc=1,
        )

        output = decoder(encoder_features, L_cond)

        assert output.shape == (B, 2, H, W)


class TestModernColorizer:
    """Test end-to-end modern colorizer."""

    @requires_hf
    def test_colorizer_initialization_classification(self):
        """Test colorizer initialization in classification mode."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="classification",
            num_classes=313,
        )

        assert model is not None
        assert model.mode == "classification"
        assert model.num_classes == 313

    @requires_hf
    def test_colorizer_initialization_regression(self):
        """Test colorizer initialization in regression mode."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="regression",
        )

        assert model is not None
        assert model.mode == "regression"

    @requires_hf
    def test_colorizer_forward_classification(self):
        """Test forward pass in classification mode."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="classification",
            num_classes=313,
        )

        B, H, W = 2, 224, 224
        L = torch.randn(B, 1, H, W)

        with torch.no_grad():
            output = model(L, return_logits=True)

        assert "ab" in output
        assert output["ab"].shape == (B, 2, H, W)
        assert "logits" in output
        assert output['logits'].shape == (B, 313, H, W)
        assert not torch.isnan(output['ab']).any()

    @requires_hf
    def test_colorizer_forward_regression(self):
        """Test forward pass in regression mode."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="regression",
        )

        B, H, W = 2, 224, 224
        L = torch.randn(B, 1, H, W)

        with torch.no_grad():
            output = model(L)

        assert "ab" in output
        assert output['ab'].shape == (B, 2, H, W)
        assert not torch.isnan(output['ab']).any()

    @requires_hf
    def test_colorizer_parameter_counting(self):
        """Test parameter counting."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="classification",
        )

        param_counts = model.count_parameters()

        assert "total" in param_counts
        assert "encoder" in param_counts
        assert "decoder" in param_counts
        assert param_counts["total"] > 0
        assert param_counts["encoder"] > 0
        assert param_counts["decoder"] > 0
        assert (
            param_counts["total"] == param_counts["encoder"] + param_counts["decoder"]
        )

    @requires_hf
    def test_colorizer_freeze_unfreeze(self):
        """Test freeze/unfreeze encoder."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="classification",
        )

        initial_counts = model.count_parameters()

        # Freeze encoder
        model.freeze_encoder()
        frozen_counts = model.count_parameters()
        assert frozen_counts["encoder_trainable"] == 0
        assert frozen_counts["decoder_trainable"] == initial_counts["decoder_trainable"]

        # Unfreeze encoder
        model.unfreeze_encoder()
        unfrozen_counts = model.count_parameters()
        assert (
            unfrozen_counts["encoder_trainable"] == initial_counts["encoder_trainable"]
        )


class TestLossFunctions:
    """Test loss functions."""

    def test_vgg_perceptual_loss(self):
        """Test VGG perceptual loss."""
        perceptual_loss = VGGPerceptualLoss(
            layers=["relu2_2", "relu3_3"],
            weights=[1.0, 1.0],
        )

        B, H, W = 2, 224, 224
        pred_rgb = torch.rand(B, 3, H, W)
        target_rgb = torch.rand(B, 3, H, W)

        loss = perceptual_loss(pred_rgb, target_rgb)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_combined_loss_classification(self):
        """Test combined loss in classification mode."""
        combined_loss = CombinedColorizationLoss(
            mode="classification",
            use_perceptual=True,
            perceptual_weight=0.1,
        )

        B, H, W = 2, 224, 224
        Q = 484  # Use actual grid size from get_ab_grid()

        pred_logits = torch.randn(B, Q, H, W)
        target_ab = torch.randn(B, 2, H, W)
        L = torch.randn(B, 1, H, W)
        target_bins = torch.randint(0, Q, (B, H, W))

        losses = combined_loss(pred_logits, target_ab, L, target_bins)

        assert "loss" in losses
        assert "classification_loss" in losses
        assert "perceptual_loss" in losses
        assert all(v.item() >= 0 for v in losses.values())

    def test_combined_loss_regression(self):
        """Test combined loss in regression mode."""
        combined_loss = CombinedColorizationLoss(
            mode="regression",
            use_perceptual=True,
            perceptual_weight=0.1,
        )

        B, H, W = 2, 224, 224

        pred_ab = torch.randn(B, 2, H, W)
        target_ab = torch.randn(B, 2, H, W)
        L = torch.randn(B, 1, H, W)

        losses = combined_loss(pred_ab, target_ab, L)

        assert "loss" in losses
        assert "regression_loss" in losses
        assert "perceptual_loss" in losses
        assert all(v.item() >= 0 for v in losses.values())


class TestGradientFlow:
    """Test gradient flow through the model."""

    @requires_hf
    def test_backward_pass(self):
        """Test gradients flow through model."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="regression",
        )

        B, H, W = 2, 224, 224
        L = torch.randn(B, 1, H, W, requires_grad=True)

        output = model(L)
        loss = output["ab"].sum()
        loss.backward()

        # Check gradients exist
        assert L.grad is not None
        assert not torch.isnan(L.grad).any()

        # Check model has gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any()

        assert has_grad, "Model should have gradients"

    @requires_hf
    def test_training_step(self):
        """Test a complete training step."""
        model = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=True,
            encoder_pretrained=True,
            mode="regression",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss()

        B, H, W = 2, 224, 224
        L = torch.randn(B, 1, H, W)
        target_ab = torch.randn(B, 2, H, W)

        # Forward
        output = model(L)
        loss = criterion(output["ab"], target_ab)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is valid
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestHFModelLoader:
    """Test HuggingFace model loader."""

    def test_loader_initialization(self):
        """Test loader can be initialized."""
        loader = HFModelLoader()
        assert loader is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
