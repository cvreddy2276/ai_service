"""Deep Volumetric Feature Extraction Module for 3D DCE-MRI Breast Cancer Analysis.

Extracts deep latent feature representations directly from 3D Convolutional Neural Networks
(3D UNet bottleneck or 3D CNN encoder) for volumetric tumor regions without classical ML.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeepVolumetricFeatureExtractor:
    """Extracts deep neural feature representations from 3D DCE-MRI volumetric tensors.

    Uses a pre-trained deep neural network encoder or feature backbone to project
    3D volumetric inputs into a compact, semantically rich latent feature embedding vector.

    Attributes:
        model: PyTorch 3D neural network module.
        device: Target compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize DeepVolumetricFeatureExtractor.

        Args:
            model: PyTorch 3D deep neural network module.
            device: Compute device (CPU or CUDA).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

    def extract_features(
        self,
        volume_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Extract deep latent feature vector from a 3D volumetric input tensor.

        Args:
            volume_tensor: 3D volumetric tensor of shape (B, C, H, W, D) or (C, H, W, D).

        Returns:
            np.ndarray: Deep latent feature vector of shape (B, feature_dim) or (feature_dim,).
        """
        if volume_tensor.ndim == 4:
            # Add batch dimension: (1, C, H, W, D)
            volume_tensor = volume_tensor.unsqueeze(0)

        volume_tensor = volume_tensor.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            # If model has a features backbone attribute (like SimpleCNN3D)
            if hasattr(self.model, "features"):
                feat_map = self.model.features(volume_tensor)
                # Flatten to 1D vector per sample
                features = feat_map.flatten(start_dim=1)
            elif hasattr(self.model, "encoder"):
                # E.g. UNet encoder bottleneck
                feat_map = self.model.encoder(volume_tensor)
                features = feat_map.flatten(start_dim=1)
            else:
                # Fallback: obtain penultimate activations via adaptive pooling
                try:
                    feat_map = list(self.model.children())[0](volume_tensor)
                    features = feat_map.flatten(start_dim=1)
                except Exception:
                    features = volume_tensor.flatten(start_dim=1)

        feat_np = features.cpu().numpy()
        logger.debug("Extracted deep feature representation with dimension %s", feat_np.shape)
        return feat_np.squeeze() if feat_np.shape[0] == 1 else feat_np
