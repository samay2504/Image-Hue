"""
ModernColorizer: End-to-end model combining transformer encoder + SPADE decoder.

Integrates ViT/Swin encoder with conditional normalization decoder for
state-of-the-art colorization.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from src.models.encoder_transformer import TransformerEncoder
from src.models.decoder_spade import SPADEDecoder
from src.models.ops import annealed_mean, ABColorQuantizer

logger = logging.getLogger(__name__)


class ModernColorizer(nn.Module):
    """
    Modern colorization pipeline with transformer encoder and SPADE decoder.
    
    Architecture:
        L_input -> TransformerEncoder -> Multi-scale features
                -> SPADEDecoder(features, L) -> Logits or ab
                -> annealed_mean (if classification) -> ab_output
    """
    
    def __init__(
        self,
        # Encoder config
        encoder_model: Optional[str] = None,
        encoder_size: str = "base",
        encoder_pretrained: bool = True,
        encoder_freeze_blocks: int = 0,
        encoder_local_only: bool = False,
        # Decoder config
        decoder_channels: List[int] = [512, 256, 128, 64],
        norm_mode: str = "spade",
        block_type: str = "convnext",
        num_blocks_per_stage: int = 2,
        # Output config
        mode: str = "classification",  # "classification" or "regression"
        num_classes: int = 313,  # Q bins for classification
        # Memory config
        use_checkpointing: bool = False,
    ):
        """
        Args:
            encoder_model: Specific HF model name
            encoder_size: Size category ("tiny", "base", "swin")
            encoder_pretrained: Use pretrained weights
            encoder_freeze_blocks: Number of encoder blocks to freeze initially
            encoder_local_only: Only use locally cached models
            decoder_channels: Channel counts for decoder stages
            norm_mode: "spade" or "adain"
            block_type: "convnext" or "residual"
            num_blocks_per_stage: Blocks per decoder stage
            mode: "classification" or "regression"
            num_classes: Number of ab bins (for classification)
            use_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        
        self.mode = mode
        self.num_classes = num_classes
        self.use_checkpointing = use_checkpointing
        
        logger.info(f"Building ModernColorizer:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Encoder: {encoder_model or encoder_size}")
        logger.info(f"  Norm: {norm_mode}, Block: {block_type}")
        
        # Encoder
        self.encoder = TransformerEncoder(
            model_name=encoder_model,
            size=encoder_size,
            pretrained=encoder_pretrained,
            use_checkpointing=use_checkpointing,
            freeze_blocks=encoder_freeze_blocks,
            local_only=encoder_local_only,
        )
        
        # Get encoder feature channels
        encoder_channels = self.encoder.get_feature_channels()
        
        # Decoder
        num_output_channels = num_classes if mode == "classification" else 2
        self.decoder = SPADEDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_output_channels=num_output_channels,
            label_nc=1,  # L channel
            norm_mode=norm_mode,
            block_type=block_type,
            num_blocks_per_stage=num_blocks_per_stage,
            use_checkpointing=use_checkpointing,
        )
        
        # Quantizer for classification mode
        if mode == "classification":
            self.quantizer = ABColorQuantizer()
        else:
            self.quantizer = None
        
        logger.info(f"Model built successfully")
        logger.info(f"  Encoder channels: {encoder_channels}")
        logger.info(f"  Decoder output: {num_output_channels} channels")
    
    def forward(
        self,
        L: torch.Tensor,
        temperature: float = 0.38,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            L: [B, 1, H, W] grayscale L channel, normalized to [-1, 1]
            temperature: Annealed-mean temperature (for classification mode)
            return_logits: Return logits in addition to ab
            
        Returns:
            Dict with keys:
                - "ab": [B, 2, H, W] predicted ab channels
                - "logits": [B, Q, H, W] (if classification and return_logits=True)
        """
        # Extract multi-scale features
        features = self.encoder(L)
        
        # Decode with conditional normalization
        output = self.decoder(features, L)
        
        result = {}
        
        if self.mode == "classification":
            # Classification: output is logits [B, Q, H, W]
            logits = output
            
            # Convert to ab using annealed-mean
            ab = annealed_mean(logits, T=temperature)
            
            result["ab"] = ab
            if return_logits:
                result["logits"] = logits
        
        else:
            # Regression: output is directly ab [B, 2, H, W]
            result["ab"] = output
        
        return result
    
    def freeze_encoder(self, freeze_blocks: Optional[int] = None):
        """
        Freeze encoder blocks.
        
        Args:
            freeze_blocks: Number of blocks to freeze (None = all)
        """
        if freeze_blocks is None:
            # Freeze entire encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze entire encoder")
        else:
            self.encoder.freeze_prefix_blocks(freeze_blocks)
    
    def unfreeze_encoder(self, last_n_blocks: Optional[int] = None):
        """
        Unfreeze encoder blocks.
        
        Args:
            last_n_blocks: Number of last blocks to unfreeze (None = all)
        """
        if last_n_blocks is None:
            self.encoder.unfreeze_all()
        else:
            self.encoder.unfreeze_last_n_blocks(last_n_blocks)
    
    def get_trainable_parameters(self) -> Dict[str, List]:
        """Get parameters grouped by component."""
        return {
            "encoder": list(self.encoder.parameters()),
            "decoder": list(self.decoder.parameters()),
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        encoder_trainable = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        decoder_trainable = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )
        
        return {
            "total": total_params,
            "encoder": encoder_params,
            "decoder": decoder_params,
            "encoder_trainable": encoder_trainable,
            "decoder_trainable": decoder_trainable,
            "total_trainable": encoder_trainable + decoder_trainable,
        }


def test_modern_colorizer():
    """Test the complete modern colorizer."""
    print("=" * 60)
    print("Testing ModernColorizer")
    print("=" * 60)
    
    try:
        # Test with classification mode
        print("\n1. Testing classification mode (Q=313):")
        model_cls = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=False,
            mode="classification",
            num_classes=313,
            norm_mode="spade",
            block_type="convnext",
        )
        
        # Count parameters
        param_counts = model_cls.count_parameters()
        print(f"   Total parameters: {param_counts['total']:,}")
        print(f"   Encoder: {param_counts['encoder']:,}")
        print(f"   Decoder: {param_counts['decoder']:,}")
        
        # Forward pass
        B, H, W = 2, 256, 256
        L = torch.randn(B, 1, H, W)
        
        print(f"\n   Forward pass with input shape: {L.shape}")
        with torch.no_grad():
            output = model_cls(L, temperature=0.38, return_logits=True)
        
        print(f"   Output ab shape: {output['ab'].shape}")
        print(f"   Output logits shape: {output['logits'].shape}")
        
        assert output['ab'].shape == (B, 2, H, W), "ab shape mismatch"
        assert output['logits'].shape == (B, 313, H, W), "logits shape mismatch"
        print("   ✓ Classification mode test passed")
        
        # Test regression mode
        print("\n2. Testing regression mode:")
        model_reg = ModernColorizer(
            encoder_size="tiny",
            encoder_local_only=False,
            mode="regression",
            norm_mode="adain",
            block_type="residual",
        )
        
        with torch.no_grad():
            output_reg = model_reg(L)
        
        print(f"   Output ab shape: {output_reg['ab'].shape}")
        assert output_reg['ab'].shape == (B, 2, H, W), "regression ab shape mismatch"
        print("   ✓ Regression mode test passed")
        
        # Test freeze/unfreeze
        print("\n3. Testing freeze/unfreeze:")
        model_cls.freeze_encoder(freeze_blocks=6)
        param_counts_frozen = model_cls.count_parameters()
        print(f"   Trainable after freeze: {param_counts_frozen['encoder_trainable']:,}")
        
        model_cls.unfreeze_encoder(last_n_blocks=2)
        param_counts_unfrozen = model_cls.count_parameters()
        print(f"   Trainable after unfreeze: {param_counts_unfrozen['encoder_trainable']:,}")
        print("   ✓ Freeze/unfreeze test passed")
        
        print("\n" + "=" * 60)
        print("✓ All ModernColorizer tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_modern_colorizer()
