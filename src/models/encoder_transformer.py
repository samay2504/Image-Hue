"""
Transformer-based encoder using Hugging Face models (ViT, Swin) with multi-scale feature extraction.

Replaces the VGG encoder with a modern transformer architecture while maintaining
compatibility with the colorization pipeline.
"""

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from src.utils.hf_model_cache import load_transformer_model

logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder wrapper for colorization.

    Extracts multi-scale features from transformer blocks and reshapes
    tokens to spatial feature maps compatible with the decoder.

    Based on ViT/Swin architectures from Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        size: str = "base",
        pretrained: bool = True,
        feature_blocks: Optional[List[int]] = None,
        use_checkpointing: bool = False,
        freeze_blocks: int = 0,
        local_only: bool = False,
    ):
        """
        Args:
            model_name: Specific HF model name (e.g., "google/vit-base-patch16-224")
            size: Size category ("tiny", "base", "swin") if model_name not specified
            pretrained: Load pretrained weights
            feature_blocks: Which transformer blocks to extract features from
                           (e.g., [3, 6, 9, 12] for 4 scales)
            use_checkpointing: Enable gradient checkpointing for memory efficiency
            freeze_blocks: Number of initial blocks to freeze
            local_only: Only use locally cached models
        """
        super().__init__()

        if not pretrained:
            raise NotImplementedError("Random initialization not supported yet")

        # Load transformer model
        logger.info(f"Loading transformer encoder (size={size})")
        self.model, self.metadata = load_transformer_model(
            model_name=model_name,
            size=size,
            local_only=local_only,
        )

        # Extract model parameters
        self.hidden_dim = self.metadata["hidden_dim"]
        self.num_blocks = self.metadata["num_blocks"]
        self.patch_size = self.metadata.get("patch_size", 16)
        self.image_size = self.metadata.get("image_size", 224)

        # Determine feature extraction blocks
        if feature_blocks is None:
            # Default: extract from 4 evenly spaced blocks
            step = max(1, self.num_blocks // 4)
            self.feature_blocks = [
                step * i for i in range(1, 5) if step * i <= self.num_blocks
            ]
        else:
            self.feature_blocks = [b for b in feature_blocks if b <= self.num_blocks]

        logger.info(f"Extracting features from blocks: {self.feature_blocks}")
        logger.info(f"Hidden dim: {self.hidden_dim}, Patch size: {self.patch_size}")

        # Enable gradient checkpointing if requested
        self.use_checkpointing = use_checkpointing
        if use_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Freeze initial blocks
        if freeze_blocks > 0:
            self.freeze_prefix_blocks(freeze_blocks)

        # Input projection (L channel to 3 channels expected by transformers)
        self.input_proj = nn.Conv2d(1, 3, kernel_size=1)

        # Feature channel dimensions for decoder (will be populated after first forward)
        self.feature_channels: List[int] = []

    def freeze_prefix_blocks(self, n: int):
        """
        Freeze the first n transformer blocks.

        Args:
            n: Number of blocks to freeze from the start
        """
        if n <= 0:
            return

        n = min(n, self.num_blocks)
        logger.info(f"Freezing first {n} transformer blocks")

        # Access blocks depending on model architecture
        if hasattr(self.model, "encoder"):
            # ViT-style
            blocks = self.model.encoder.layer[:n]
        elif hasattr(self.model, "layers"):
            # Swin-style
            blocks = self.model.layers[:n]
        else:
            logger.warning("Unknown model architecture, cannot freeze blocks")
            return

        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All transformer blocks unfrozen")

    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreeze the last n transformer blocks.

        Args:
            n: Number of blocks to unfreeze from the end
        """
        if n <= 0:
            return

        n = min(n, self.num_blocks)
        logger.info(f"Unfreezing last {n} transformer blocks")

        # Access blocks depending on model architecture
        if hasattr(self.model, "encoder"):
            blocks = self.model.encoder.layer[-n:]
        elif hasattr(self.model, "layers"):
            blocks = self.model.layers[-n:]
        else:
            logger.warning("Unknown model architecture, cannot unfreeze blocks")
            return

        for block in blocks:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, L: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass extracting multi-scale features.

        Args:
            L: Grayscale input, shape [B, 1, H, W]

        Returns:
            List of feature tensors [f1, f2, f3, f4] with shapes:
            - f1: [B, C1, H1, W1] (earliest/lowest resolution)
            - f2: [B, C2, H2, W2]
            - f3: [B, C3, H3, W3]
            - f4: [B, C4, H4, W4] (latest/highest resolution)

            Where H_i/W_i increase and C_i are transformer hidden dims.
        """
        B, _, H, W = L.shape

        # Project L to 3 channels (RGB-like input for transformers)
        x = self.input_proj(L)  # [B, 3, H, W]

        # Resize if needed to match expected input size
        if H != self.image_size or W != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Forward through transformer and collect intermediate features
        features = []

        # Get transformer outputs with intermediate hidden states
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # Tuple of [B, N_tokens, C]

        # Extract features from specified blocks
        H_patch = self.image_size // self.patch_size
        W_patch = self.image_size // self.patch_size

        for block_idx in self.feature_blocks:
            if block_idx < len(hidden_states):
                tokens = hidden_states[block_idx]  # [B, N_tokens, C]

                # Reshape tokens to spatial feature map
                # tokens: [B, N_tokens, C] -> [B, C, H_patch, W_patch]
                feature_map = self._tokens_to_feature_map(tokens, H_patch, W_patch)

                features.append(feature_map)

        # Resize features back to match input resolution
        features_resized = []
        target_sizes = [
            (H // 16, W // 16),  # 1/16 resolution
            (H // 8, W // 8),  # 1/8 resolution
            (H // 4, W // 4),  # 1/4 resolution
            (H // 2, W // 2),  # 1/2 resolution
        ]

        for i, feat in enumerate(features):
            if i < len(target_sizes):
                h, w = target_sizes[i]
                feat_resized = torch.nn.functional.interpolate(
                    feat, size=(h, w), mode="bilinear", align_corners=False
                )
                features_resized.append(feat_resized)

        # Pad if we don't have enough features
        while len(features_resized) < 4:
            features_resized.append(features_resized[-1])

        # Update channel dimensions
        if not self.feature_channels:
            self.feature_channels = [f.shape[1] for f in features_resized]
            logger.info(f"Feature channels: {self.feature_channels}")

        return features_resized

    def _tokens_to_feature_map(
        self, tokens: torch.Tensor, H_patch: int, W_patch: int
    ) -> torch.Tensor:
        """
        Reshape token sequence to spatial feature map.

        Args:
            tokens: [B, N_tokens, C] where N_tokens = H_patch * W_patch (+ cls token)
            H_patch: Height in patches
            W_patch: Width in patches

        Returns:
            feature_map: [B, C, H_patch, W_patch]
        """
        B, N, C = tokens.shape

        # Handle CLS token if present
        if N == H_patch * W_patch + 1:
            # Remove CLS token (first token)
            tokens = tokens[:, 1:, :]  # [B, H*W, C]
        elif N != H_patch * W_patch:
            # Adjust to match expected dimensions
            logger.warning(
                f"Token count mismatch: {N} vs {H_patch}*{W_patch}={H_patch*W_patch}"
            )
            # Try to interpolate
            tokens = tokens[:, : H_patch * W_patch, :]

        # Reshape to spatial
        feature_map = tokens.permute(0, 2, 1)  # [B, C, N]
        feature_map = feature_map.reshape(B, C, H_patch, W_patch)

        return feature_map

    def get_feature_channels(self) -> List[int]:
        """Get the number of channels in each feature level."""
        if not self.feature_channels:
            # Run dummy forward to populate
            dummy = torch.zeros(1, 1, self.image_size, self.image_size)
            with torch.no_grad():
                features = self.forward(dummy)
            self.feature_channels = [f.shape[1] for f in features]

        return self.feature_channels


def test_encoder():
    """Test the encoder with a dummy input."""
    print("Testing TransformerEncoder...")

    # Try to load a model
    try:
        encoder = TransformerEncoder(size="tiny", local_only=False)

        # Test forward pass
        dummy_input = torch.randn(2, 1, 256, 256)
        features = encoder(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        for i, feat in enumerate(features):
            print(f"Feature {i+1} shape: {feat.shape}")

        print("✓ Encoder test passed!")

    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_encoder()
