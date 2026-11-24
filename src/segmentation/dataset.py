"""Dataset and transformation pipelines for 3D tumor segmentation with MONAI.

Handles paired subtraction volumes and ground-truth lesion masks with
class-balanced positive/negative patch sampling to solve severe tumor volume imbalance.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
)

logger = logging.getLogger(__name__)


def get_transforms(
    keys: Sequence[str] = ("image", "label"),
    stage: str = "train",
    spatial_size: Tuple[int, int, int] = (96, 96, 32),
    num_samples: int = 2,
    pos: float = 1.0,
    neg: float = 1.0,
) -> Compose:
    """Build MONAI transform pipeline for 3D tumor segmentation.

    Args:
        keys: Dictionary keys representing (image, label).
        stage: Pipeline stage ('train', 'val', or 'test').
        spatial_size: Crop ROI dimensions (H, W, D) for patch sampling (default: (96, 96, 32)).
        num_samples: Number of positive/negative patches sampled per volume in training.
        pos: Relative sampling weight for foreground (tumor) voxels.
        neg: Relative sampling weight for background voxels.

    Returns:
        Compose: Composed MONAI transforms.
    """
    image_key = keys[0]
    label_key = keys[1]
    all_keys = list(keys)

    transforms_list: List[Callable] = [
        # 1. Load NIfTI and set channel first format (1, H, W, D)
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=False, allow_missing_keys=True),
        # 2. Canonical anatomical orientation
        Orientationd(keys=all_keys, axcodes="RAS", allow_missing_keys=True),
        # 3. Non-zero intensity standardization on input image
        NormalizeIntensityd(keys=[image_key], nonzero=True, channel_wise=True),
    ]

    # 4. Training-specific augmentations and positive/negative patch extraction
    if stage.lower() == "train":
        transforms_list.extend([
            # Solves severe foreground/background class imbalance
            RandCropByPosNegLabeld(
                keys=all_keys,
                label_key=label_key,
                spatial_size=spatial_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key=image_key,
            ),
            # 3D spatial rotations
            RandRotated(
                keys=all_keys,
                range_x=0.3,
                range_y=0.3,
                range_z=0.3,
                prob=0.5,
                mode=("bilinear", "nearest"),
            ),
            # 3D spatial flips along H, W, D
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=[0, 1, 2]),
        ])

    # 5. Ensure PyTorch float32 tensor type
    transforms_list.extend([
        CastToTyped(keys=all_keys, dtype=torch.float32),
        EnsureTyped(keys=all_keys),
    ])

    return Compose(transforms_list)


def get_segmentation_dataloader(
    data: Sequence[Dict[str, Any]],
    stage: str = "train",
    batch_size: int = 2,
    spatial_size: Tuple[int, int, int] = (96, 96, 32),
    num_samples: int = 2,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    use_cache: bool = False,
    cache_rate: float = 1.0,
) -> DataLoader:
    """Create a PyTorch DataLoader for 3D tumor segmentation.

    Args:
        data: List of sample dictionaries containing 'image' and 'label'.
        stage: Pipeline stage ('train', 'val', 'test').
        batch_size: Mini-batch size.
        spatial_size: Patch size for training.
        num_samples: Number of patches per volume in training.
        num_workers: Number of DataLoader worker subprocesses.
        shuffle: Whether to shuffle data (default: True for train, False otherwise).
        use_cache: Whether to use MONAI CacheDataset.
        cache_rate: Fraction of dataset cached in RAM if use_cache=True.

    Returns:
        DataLoader: Configured MONAI/PyTorch DataLoader.
    """
    is_train = stage.lower() == "train"
    if shuffle is None:
        shuffle = is_train

    transforms = get_transforms(
        stage=stage,
        spatial_size=spatial_size,
        num_samples=num_samples if is_train else 1,
    )

    if use_cache:
        dataset = CacheDataset(
            data=data,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        dataset = Dataset(data=data, transform=transforms)

    logger.debug(
        "Initialized segmentation DataLoader: %d samples, stage=%s, batch_size=%d",
        len(data),
        stage,
        batch_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
