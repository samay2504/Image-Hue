"""
Hugging Face model cache management with fallback strategies.

Safe loader for transformer models with automatic fallback to smaller variants
and local-first loading strategy.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch

logger = logging.getLogger(__name__)

# HF cache location
HF_LOCAL_CACHE = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()

# Model candidates in order of preference (larger -> smaller)
# Format: (model_name, size_category, expected_hidden_dim, expected_num_blocks)
MODEL_CANDIDATES = [
    ("google/vit-base-patch16-224", "base", 768, 12),
    ("facebook/swin-base-patch4-window7-224", "swin-base", 768, 12),
    ("google/vit-tiny-patch16-224-in21k", "tiny", 192, 12),
]

# Size category mapping
SIZE_TO_MODELS = {
    "tiny": ["google/vit-tiny-patch16-224-in21k"],
    "base": ["google/vit-base-patch16-224", "facebook/swin-base-patch4-window7-224"],
    "swin": ["facebook/swin-base-patch4-window7-224"],
}


class HFModelLoader:
    """Safe loader for Hugging Face transformer models."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: Override default HF cache directory
        """
        self.cache_dir = cache_dir or HF_LOCAL_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"HF cache directory: {self.cache_dir}")

    def load_model(
        self,
        model_name: Optional[str] = None,
        size: str = "base",
        local_only: bool = False,
        fallback_on_error: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a transformer model with fallback strategy.

        Args:
            model_name: Specific model name to load (e.g., "google/vit-base-patch16-224")
            size: Size category ("tiny", "base", "swin") if model_name not specified
            local_only: Only use locally cached models
            fallback_on_error: Try smaller models on failure

        Returns:
            (model, metadata) where metadata contains:
                - "model_name": str
                - "size": str
                - "source": "local" | "downloaded" | "fallback"
                - "hidden_dim": int
                - "num_blocks": int
        """
        # Determine candidates to try
        if model_name:
            candidates = [(model_name, "custom", None, None)]
        else:
            candidates = [
                (name, sz, dim, blocks)
                for name, sz, dim, blocks in MODEL_CANDIDATES
                if size == "tiny"
                or sz != "tiny"  # Skip tiny unless explicitly requested
            ]

            # Filter by size preference
            if size in SIZE_TO_MODELS:
                preferred = SIZE_TO_MODELS[size]
                candidates = [
                    c for c in candidates if c[0] in preferred
                ] + candidates  # Preferred first, then fallbacks

        last_error = None

        for candidate_name, candidate_size, hidden_dim, num_blocks in candidates:
            try:
                logger.info(f"Attempting to load model: {candidate_name}")

                # Try local-only first
                try:
                    model, metadata = self._load_single_model(
                        candidate_name, local_files_only=True
                    )
                    metadata.update(
                        {
                            "model_name": candidate_name,
                            "size": candidate_size,
                            "source": "local",
                            "hidden_dim": hidden_dim or metadata.get("hidden_dim"),
                            "num_blocks": num_blocks or metadata.get("num_blocks"),
                        }
                    )
                    logger.info(f"✓ Loaded from local cache: {candidate_name}")
                    return model, metadata

                except Exception as e:
                    if local_only:
                        raise
                    logger.debug(f"Not in local cache: {candidate_name}")

                # Try downloading if not local_only
                if not local_only:
                    logger.info(f"Downloading model: {candidate_name}")
                    model, metadata = self._load_single_model(
                        candidate_name, local_files_only=False
                    )
                    metadata.update(
                        {
                            "model_name": candidate_name,
                            "size": candidate_size,
                            "source": "downloaded",
                            "hidden_dim": hidden_dim or metadata.get("hidden_dim"),
                            "num_blocks": num_blocks or metadata.get("num_blocks"),
                        }
                    )
                    logger.info(f"✓ Downloaded and loaded: {candidate_name}")
                    return model, metadata

            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load {candidate_name}: {e}")

                if not fallback_on_error:
                    raise

                # Try next candidate
                continue

        # All candidates failed
        error_msg = f"Failed to load any model candidate. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _load_single_model(
        self, model_name: str, local_files_only: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a single model from HuggingFace.

        Returns:
            (model, metadata_dict)
        """
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )

        # Load config first to get metadata
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
        )

        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
            add_pooling_layer=False,  # We'll use intermediate features
        )

        # Extract metadata
        metadata = {
            "config": config,
            "hidden_dim": getattr(config, "hidden_size", None),
            "num_blocks": getattr(config, "num_hidden_layers", None),
            "patch_size": getattr(config, "patch_size", 16),
            "image_size": getattr(config, "image_size", 224),
        }

        return model, metadata

    def list_available_local_models(self) -> List[str]:
        """List models available in local cache."""
        available = []

        for model_name, _, _, _ in MODEL_CANDIDATES:
            try:
                # Check if model exists locally
                model_path = self.cache_dir / "models" / model_name.replace("/", "--")
                if model_path.exists():
                    available.append(model_name)
            except Exception:
                continue

        return available

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get info about a model candidate."""
        for name, size, dim, blocks in MODEL_CANDIDATES:
            if name == model_name:
                return {
                    "name": name,
                    "size": size,
                    "hidden_dim": dim,
                    "num_blocks": blocks,
                }
        return None


def get_available_models(local_only: bool = False) -> List[Dict[str, Any]]:
    """
    Get list of available models.

    Args:
        local_only: Only return locally cached models

    Returns:
        List of model info dicts
    """
    loader = HFModelLoader()

    if local_only:
        available_names = loader.list_available_local_models()
        return [
            loader.get_model_info(name)
            for name in available_names
            if loader.get_model_info(name)
        ]
    else:
        return [
            {
                "name": name,
                "size": size,
                "hidden_dim": dim,
                "num_blocks": blocks,
            }
            for name, size, dim, blocks in MODEL_CANDIDATES
        ]


def load_transformer_model(
    model_name: Optional[str] = None,
    size: str = "base",
    local_only: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to load a transformer model.

    Args:
        model_name: Specific model name
        size: Size category if model_name not specified
        local_only: Only use local cache
        device: Device to move model to

    Returns:
        (model, metadata)
    """
    loader = HFModelLoader()
    model, metadata = loader.load_model(
        model_name=model_name,
        size=size,
        local_only=local_only,
    )

    if device is not None:
        model = model.to(device)

    return model, metadata
