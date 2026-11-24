"""3D Deep Convolutional Neural Network Architectures for Breast Cancer DCE-MRI Classification.

Provides volumetric deep learning backbones (3D DenseNet, 3D ResNet, and Lightweight 3D CNN)
to predict treatment response (pathological Complete Response, pCR) directly from 3D/4D DCE-MRI volumes.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, DenseNet169, resnet18, resnet34

logger = logging.getLogger(__name__)


class SimpleCNN3D(nn.Module):
    """Lightweight 3D Convolutional Neural Network for Volumetric DCE-MRI Classification.

    Designed for fast convergence and memory-constrained training on 3D medical volumes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initialize SimpleCNN3D.

        Args:
            in_channels: Number of input channels (e.g., 1 for subtraction, 3 for multi-channel DCE).
            num_classes: Number of output logits (1 for binary pCR prediction).
            dropout: Dropout probability before the classification head.
        """
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),
            # Block 2
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),
            # Block 3
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),
            # Block 4
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input 3D volumetric tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        feat = self.features(x)
        return self.classifier(feat)


def build_3d_classifier(
    spatial_dims: int = 3,
    in_channels: int = 1,
    num_classes: int = 1,
    backbone: str = "densenet121",
    dropout: float = 0.2,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Build and initialize a 3D Deep Learning classifier for DCE-MRI treatment response prediction.

    Supported backbones:
      - 'densenet121': MONAI 3D DenseNet-121.
      - 'densenet169': MONAI 3D DenseNet-169.
      - 'resnet18': MONAI 3D ResNet-18.
      - 'resnet34': MONAI 3D ResNet-34.
      - 'simple_cnn3d': Fast lightweight 4-stage 3D CNN.

    Args:
        spatial_dims: Number of spatial dimensions (3 for volumetric MRI).
        in_channels: Number of input channels (1 for subtraction, 3 for multi-channel DCE).
        num_classes: Number of output classes/logits (default: 1 for binary classification).
        backbone: Model backbone architecture name (default: 'densenet121').
        dropout: Dropout rate in classifier head.
        device: Target PyTorch device (CPU or CUDA).

    Returns:
        nn.Module: Instantiated 3D deep neural network model.

    Raises:
        ValueError: If spatial_dims != 3 or backbone name is unrecognized.
    """
    if spatial_dims != 3:
        raise ValueError(f"Expected spatial_dims=3 for 3D DCE-MRI, got {spatial_dims}")

    backbone_lower = backbone.lower().strip()
    logger.info(
        "Building 3D Deep Classifier: backbone='%s', in_channels=%d, num_classes=%d",
        backbone_lower,
        in_channels,
        num_classes,
    )

    if backbone_lower == "densenet121":
        model: nn.Module = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=dropout,
        )
    elif backbone_lower == "densenet169":
        model = DenseNet169(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=dropout,
        )
    elif backbone_lower == "resnet18":
        model = resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
        )
    elif backbone_lower == "resnet34":
        model = resnet34(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
        )
    elif backbone_lower in ("simple_cnn3d", "simple3d", "cnn3d"):
        model = SimpleCNN3D(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        supported = ["densenet121", "densenet169", "resnet18", "resnet34", "simple_cnn3d"]
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: {supported}")

    if device is not None:
        model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "3D Deep Classifier initialized: %d trainable parameters (%.2f MB)",
        param_count,
        param_count * 4 / (1024 ** 2),
    )

    return model
