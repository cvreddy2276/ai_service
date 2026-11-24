"""3D UNet Model Architecture for 3D Breast Tumor Segmentation.

Implements deep 3D residual UNet models using MONAI framework.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import UNet

logger = logging.getLogger(__name__)


def build_3d_unet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: Union[str, Tuple] = "instance",
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> UNet:
    """Build and initialize a deep 3D Residual UNet for tumor segmentation.

    Args:
        spatial_dims: Number of spatial dimensions (3 for volumetric MRI).
        in_channels: Number of input channels (1 for subtraction volume I_sub).
        out_channels: Number of output segmentation classes (1 for binary tumor mask).
        channels: Sequence of feature channels at each resolution level (default: (16, 32, 64, 128, 256)).
        strides: Convolution downsampling strides (default: (2, 2, 2, 2)).
        num_res_units: Number of residual blocks per resolution level (default: 2).
        norm: Normalization type ('instance', 'batch', etc.) (default: 'instance').
        dropout: Dropout probability (default: 0.1).
        device: Target PyTorch device (CPU/CUDA).

    Returns:
        UNet: Configured MONAI 3D UNet neural network module.

    Raises:
        ValueError: If spatial_dims is not 3 or channel/stride specifications are invalid.
    """
    if spatial_dims != 3:
        raise ValueError(f"Expected spatial_dims=3 for 3D DCE-MRI, got {spatial_dims}")
    if len(strides) != len(channels) - 1:
        raise ValueError(
            f"Strides count ({len(strides)}) must equal channels count minus 1 ({len(channels) - 1})"
        )

    logger.info(
        "Building 3D UNet: in_channels=%d, out_channels=%d, channels=%s, res_units=%d",
        in_channels,
        out_channels,
        channels,
        num_res_units,
    )

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("3D UNet initialized with %d trainable parameters (%.2f MB)", total_params, total_params * 4 / (1024 ** 2))

    return model
