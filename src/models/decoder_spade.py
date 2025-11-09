"""
Decoder with SPADE/AdaIN normalization and ConvNeXt/Residual blocks.

Combines transformer encoder features with conditional normalization for
high-quality colorization output.
"""

import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.normalization import ConditionalNorm2d

logger = logging.getLogger(__name__)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style residual block with SPADE/AdaIN modulation.

    Based on "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
    """

    def __init__(
        self,
        dim: int,
        label_nc: int = 1,
        norm_mode: str = "spade",
        drop_path: float = 0.0,
    ):
        """
        Args:
            dim: Number of input/output channels
            label_nc: Number of conditioning channels
            norm_mode: "spade" or "adain"
            drop_path: Stochastic depth rate
        """
        super().__init__()

        # Depthwise conv (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Conditional normalization
        self.norm = ConditionalNorm2d(dim, label_nc=label_nc, mode=norm_mode)

        # Pointwise/Inverted bottleneck
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            cond: [B, label_nc, H, W] conditioning map

        Returns:
            out: [B, C, H, W]
        """
        shortcut = x

        # Depthwise conv
        x = self.dwconv(x)

        # Conditional norm
        x = self.norm(x, cond)

        # Pointwise (channel-first to channel-last)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Residual + stochastic depth
        x = shortcut + self.drop_path(x)

        return x


class ResidualBlock(nn.Module):
    """
    Standard residual block with conditional normalization.
    """

    def __init__(
        self,
        dim: int,
        label_nc: int = 1,
        norm_mode: str = "spade",
    ):
        """
        Args:
            dim: Number of channels
            label_nc: Number of conditioning channels
            norm_mode: "spade" or "adain"
        """
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = ConditionalNorm2d(dim, label_nc=label_nc, mode=norm_mode)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = ConditionalNorm2d(dim, label_nc=label_nc, mode=norm_mode)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            cond: [B, label_nc, H, W]

        Returns:
            out: [B, C, H, W]
        """
        shortcut = x

        x = self.conv1(x)
        x = self.norm1(x, cond)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x, cond)

        x = x + shortcut
        x = self.act2(x)

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SPADEDecoder(nn.Module):
    """
    Decoder with SPADE/AdaIN normalization and skip connections.

    Takes multi-scale features from transformer encoder and generates
    colorization output with conditional normalization.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_output_channels: int = 313,  # Q bins or 2 for ab
        label_nc: int = 1,
        norm_mode: str = "spade",
        block_type: str = "convnext",
        num_blocks_per_stage: int = 2,
        use_checkpointing: bool = False,
    ):
        """
        Args:
            encoder_channels: List of channel counts from encoder features [C1, C2, C3, C4]
            decoder_channels: List of channel counts for decoder stages
            num_output_channels: Output channels (Q=313 for classification, 2 for regression)
            label_nc: Channels in conditioning map (1 for L channel)
            norm_mode: "spade" or "adain"
            block_type: "convnext" or "residual"
            num_blocks_per_stage: Number of blocks per decoder stage
            use_checkpointing: Enable gradient checkpointing
        """
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_output_channels = num_output_channels
        self.label_nc = label_nc
        self.norm_mode = norm_mode
        self.use_checkpointing = use_checkpointing

        logger.info(f"Building SPADE decoder:")
        logger.info(f"  Encoder channels: {encoder_channels}")
        logger.info(f"  Decoder channels: {decoder_channels}")
        logger.info(f"  Output channels: {num_output_channels}")
        logger.info(f"  Norm mode: {norm_mode}")
        logger.info(f"  Block type: {block_type}")

        # Lateral connections from encoder features
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(enc_ch, dec_ch, kernel_size=1)
                for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
            ]
        )

        # Decoder stages
        self.decoder_stages = nn.ModuleList()

        for i, dec_ch in enumerate(decoder_channels):
            stage_blocks = []

            for j in range(num_blocks_per_stage):
                if block_type == "convnext":
                    block = ConvNeXtBlock(
                        dec_ch, label_nc=label_nc, norm_mode=norm_mode
                    )
                else:
                    block = ResidualBlock(
                        dec_ch, label_nc=label_nc, norm_mode=norm_mode
                    )
                stage_blocks.append(block)

            self.decoder_stages.append(nn.ModuleList(stage_blocks))

        # Upsample layers
        self.upsample_layers = nn.ModuleList(
            [
                (
                    nn.Sequential(
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        ),
                        nn.Conv2d(
                            decoder_channels[i],
                            decoder_channels[i + 1],
                            kernel_size=3,
                            padding=1,
                        ),
                    )
                    if i < len(decoder_channels) - 1
                    else nn.Identity()
                )
                for i in range(len(decoder_channels))
            ]
        )

        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(
                decoder_channels[-1],
                decoder_channels[-1] // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1] // 2, num_output_channels, kernel_size=1),
        )

    def forward(
        self, encoder_features: List[torch.Tensor], L_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: List of [B, C_i, H_i, W_i] from encoder
            L_cond: [B, 1, H, W] L channel for conditioning

        Returns:
            output: [B, num_output_channels, H, W]
        """
        # Process features through lateral connections
        x = self.lateral_convs[-1](encoder_features[-1])  # Start from deepest features

        # Decoder stages with skip connections (bottom-up)
        for i in range(len(self.decoder_channels) - 1, -1, -1):
            # Apply decoder blocks with conditional norm
            for block in self.decoder_stages[i]:
                if self.use_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, L_cond)
                else:
                    x = block(x, L_cond)

            # Upsample and fuse with skip connection
            if i > 0:
                x = self.upsample_layers[i](x)

                # Add skip connection from encoder
                skip = self.lateral_convs[i - 1](encoder_features[i - 1])

                # Resize if needed
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(
                        x, size=skip.shape[2:], mode="bilinear", align_corners=False
                    )

                x = x + skip

        # Final upsample to match input resolution
        if x.shape[2:] != L_cond.shape[2:]:
            x = F.interpolate(
                x, size=L_cond.shape[2:], mode="bilinear", align_corners=False
            )

        # Output head
        output = self.output_conv(x)

        return output


def test_decoder():
    """Test the SPADE decoder."""
    print("Testing SPADEDecoder...")

    B, H, W = 2, 256, 256
    encoder_channels = [192, 384, 512, 768]

    # Simulate encoder features at different scales
    encoder_features = [
        torch.randn(B, encoder_channels[0], H // 16, W // 16),
        torch.randn(B, encoder_channels[1], H // 8, W // 8),
        torch.randn(B, encoder_channels[2], H // 4, W // 4),
        torch.randn(B, encoder_channels[3], H // 2, W // 2),
    ]

    L_cond = torch.randn(B, 1, H, W)

    # Test with classification head
    print("\n1. Testing with classification head (Q=313):")
    decoder_cls = SPADEDecoder(
        encoder_channels=encoder_channels,
        num_output_channels=313,
        norm_mode="spade",
        block_type="convnext",
    )

    output_cls = decoder_cls(encoder_features, L_cond)
    print(f"   Output shape: {output_cls.shape}")
    assert output_cls.shape == (B, 313, H, W), "Classification output shape mismatch"
    print("   ✓ Classification head test passed")

    # Test with regression head
    print("\n2. Testing with regression head (ab channels):")
    decoder_reg = SPADEDecoder(
        encoder_channels=encoder_channels,
        num_output_channels=2,
        norm_mode="adain",
        block_type="residual",
    )

    output_reg = decoder_reg(encoder_features, L_cond)
    print(f"   Output shape: {output_reg.shape}")
    assert output_reg.shape == (B, 2, H, W), "Regression output shape mismatch"
    print("   ✓ Regression head test passed")

    print("\n✓ All decoder tests passed!")


if __name__ == "__main__":
    test_decoder()
