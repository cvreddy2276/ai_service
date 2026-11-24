"""Local dataset loader and transform pipeline for 3D DCE-MRI Breast Cancer data.

Parses multi-temporal DCE-MRI sequences (T0 pre-contrast, T1 post-contrast)
and ground-truth tumor masks from the BreastDCEDL I-SPY1 dataset structure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    Spacingd,
)

from src.preprocessing.transforms import (
    ComputeSubtractiond,
    N4BiasFieldCorrectiond,
    TumorRoiCropd,
    ZScoreNormalizeIntensityd,
)

logger = logging.getLogger(__name__)


def build_dce_transforms(
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    roi_size: Sequence[int] = (128, 128, 64),
    is_training: bool = False,
    apply_n4: bool = False,
    clip_subtraction: bool = True,
    keys: Sequence[str] = ("t0", "t1", "mask"),
) -> Compose:
    """Build a MONAI transform pipeline for 3D multi-temporal DCE-MRI.

    Pipeline stages:
    1. Load NIfTI volumes (T0, T1, Mask).
    2. Ensure channel-first format (1, H, W, D).
    3. Reorient to canonical spatial orientation (RAS).
    4. Resample voxels to isotropic spacing (default: 1.0 x 1.0 x 1.0 mm).
    5. Optional SimpleITK N4 bias field correction.
    6. Compute post-minus-pre subtraction as the single model input channel.
    7. Z-score intensity normalization over non-zero tissue voxels.
    8. Center-crop around tumor ROI (default: 128 x 128 x 64).
    9. Data augmentations (if is_training=True): 3D random flips, rotations, noise.
    10. Ensure PyTorch float32 / uint8 tensor formats.

    Args:
        pixdim: Target voxel spacing in millimeters (default: (1.0, 1.0, 1.0)).
        roi_size: Spatial crop dimensions (default: (128, 128, 64)).
        is_training: Whether to include data augmentation transforms.
        apply_n4: Whether to apply N4 bias field correction.
        clip_subtraction: Whether to clamp negative subtraction values to 0.
        keys: Base dictionary keys for input files.

    Returns:
        Compose: Composed MONAI transform pipeline.
    """
    image_keys = [k for k in keys if k != "mask"]
    all_keys = list(keys)

    transforms_list: List[Callable] = [
        # 1. Load NIfTI files
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=False),
        # 2. Reorient to standard anatomical coordinates (RAS)
        Orientationd(keys=all_keys, axcodes="RAS"),
        # 3. Resample to isotropic spacing
        Spacingd(
            keys=all_keys,
            pixdim=pixdim,
            mode=["bilinear" if k in image_keys else "nearest" for k in all_keys],
        ),
    ]

    # 4. Optional N4 Bias Field Correction
    if apply_n4:
        transforms_list.append(
            N4BiasFieldCorrectiond(keys=image_keys, mask_key="mask" if "mask" in all_keys else None)
        )

    # 5. Compute subtraction volume (T1 - T0) and multi-channel DCE tensor (T0, T1, Sub)
    transforms_list.append(
        ComputeSubtractiond(
            pre_key="t0",
            post_key="t1",
            sub_key="subtraction",
            multichannel_key="image",
            stack_phases=False,
            clip_negative=clip_subtraction,
        )
    )

    # 6. Intensity normalization (non-zero foreground Z-score)
    norm_keys = list(image_keys) + ["subtraction", "image"]
    transforms_list.append(
        ZScoreNormalizeIntensityd(keys=norm_keys, channel_wise=True, nonzero=True)
    )

    # 7. Tumor ROI Cropping to fixed target size (128, 128, 64)
    crop_keys = ["image", "t0", "t1", "subtraction", "mask"]
    transforms_list.append(
        TumorRoiCropd(
            keys=crop_keys,
            roi_size=roi_size,
            # Classification must not use the ground-truth mask to choose the input crop.
            mask_key=None,
            allow_missing_keys=True,
        )
    )

    # 8. Training augmentations
    if is_training:
        transforms_list.extend([
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        ])

    # 9. Type casting
    transforms_list.extend([
        CastToTyped(keys=["image", "t0", "t1", "subtraction"], dtype=torch.float32),
        CastToTyped(keys=["mask"], dtype=torch.float32 if "mask" in all_keys else None, allow_missing_keys=True),
        EnsureTyped(keys=crop_keys, allow_missing_keys=True),
    ])

    return Compose(transforms_list)


class BreastDCEDataset(Dataset):
    """Dataset for 3D DCE-MRI breast cancer volumes and clinical annotations.

    Compatible with the BreastDCEDL I-SPY1 dataset layout:
    - Pre-contrast (T0): `<PID>_spy1_vis1_acq0.nii.gz`
    - Post-contrast (T1): `<PID>_spy1_vis1_acq1.nii.gz`
    - Tumor Mask: `<PID>_spy1_vis1_mask.nii.gz`
    - Clinical metadata: `BreastDCEDL_spy1_metadata.csv` (pCR, ER, PR, HER2, etc.)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_csv: Optional[Union[str, Path]] = None,
        patient_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        target_column: str = "pCR",
        dce_folder_name: str = "spt1_dce",
        mask_folder_name: str = "spy1_mask",
    ) -> None:
        """Initialize BreastDCEDataset.

        Args:
            data_dir: Root dataset directory containing DCE and mask subfolders.
            metadata_csv: Optional path to metadata CSV file.
            patient_ids: Optional subset of patient IDs to include.
            transform: MONAI transform pipeline to apply.
            target_column: Metadata target column for supervision (e.g. 'pCR').
            dce_folder_name: Subfolder name for DCE NIfTI volumes.
            mask_folder_name: Subfolder name for mask NIfTI volumes.

        Raises:
            FileNotFoundError: If data directory does not exist.
            ValueError: If no valid patient samples with both DCE and mask are found.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.target_column = target_column
        self.dce_dir = self.data_dir / dce_folder_name
        self.mask_dir = self.data_dir / mask_folder_name

        # Fallback if raw/flat structure
        if not self.dce_dir.exists():
            self.dce_dir = self.data_dir
        if not self.mask_dir.exists():
            self.mask_dir = self.data_dir

        # Load metadata if available
        self.metadata_df: Optional[pd.DataFrame] = None
        if metadata_csv is not None and Path(metadata_csv).exists():
            self.metadata_df = pd.read_csv(metadata_csv)
            required_columns = {"pid", self.target_column}
            missing_columns = required_columns.difference(self.metadata_df.columns)
            if missing_columns:
                raise ValueError(
                    f"Metadata CSV {metadata_csv} is missing required columns: "
                    f"{sorted(missing_columns)}."
                )
            self.metadata_df["pid"] = self.metadata_df["pid"].astype(str)
            self.metadata_df.set_index("pid", inplace=True)

        self.samples = self._discover_samples(patient_ids)
        if len(self.samples) == 0:
            raise ValueError(f"No valid patient samples found in {self.data_dir}")

        logger.info(
            "Initialized BreastDCEDataset with %d subjects from %s",
            len(self.samples),
            self.data_dir,
        )
        super().__init__(data=self.samples, transform=transform)

    def _discover_samples(self, patient_ids: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        """Discover valid patient pairs (T0, T1, mask) from directory."""
        samples: List[Dict[str, Any]] = []

        # Find all acq0 (pre-contrast) files
        acq0_files = list(self.dce_dir.glob("*_acq0.nii*")) + list(self.dce_dir.glob("*_pre.nii*"))

        for pre_path in sorted(acq0_files):
            # Parse patient ID from filename (e.g. 'ISPY1_1001_spy1_vis1_acq0.nii.gz' -> 'ISPY1_1001')
            filename = pre_path.name
            pid = filename.split("_spy1")[0] if "_spy1" in filename else filename.split("_acq0")[0].split("_pre")[0]

            if patient_ids is not None and pid not in patient_ids:
                continue

            # Look for matching T1 (acq1)
            t1_name = filename.replace("_acq0", "_acq1").replace("_pre", "_post")
            t1_path = self.dce_dir / t1_name
            if not t1_path.exists():
                # Try glob fallback
                t1_candidates = list(self.dce_dir.glob(f"{pid}*acq1.nii*")) + list(self.dce_dir.glob(f"{pid}*post.nii*"))
                if len(t1_candidates) > 0:
                    t1_path = t1_candidates[0]
                else:
                    logger.warning("Skipping %s: matching T1 post-contrast volume not found.", pid)
                    continue

            # Look for matching Mask
            mask_name = filename.replace("_acq0", "_mask").replace("_pre", "_mask")
            mask_path = self.mask_dir / mask_name
            if not mask_path.exists():
                mask_candidates = list(self.mask_dir.glob(f"{pid}*mask.nii*"))
                if len(mask_candidates) > 0:
                    mask_path = mask_candidates[0]
                else:
                    logger.warning("Skipping %s: matching tumor mask not found.", pid)
                    continue

            # Parse clinical metadata and label
            if self.metadata_df is None or pid not in self.metadata_df.index:
                logger.warning("Skipping %s: no clinical metadata row is available.", pid)
                continue
            row = self.metadata_df.loc[pid]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Duplicate metadata rows found for patient {pid}.")
            if self.target_column not in row or pd.isna(row[self.target_column]):
                logger.warning("Skipping %s: outcome '%s' is missing.", pid, self.target_column)
                continue
            label = float(row[self.target_column])
            if label not in (0.0, 1.0):
                raise ValueError(f"Patient {pid} has non-binary {self.target_column} value {label}.")
            samples.append({
                "pid": pid,
                "t0": str(pre_path),
                "t1": str(t1_path),
                "mask": str(mask_path),
                "label": torch.tensor(label, dtype=torch.float32),
            })

        return samples


def get_dce_dataloader(
    data_dir: Union[str, Path],
    metadata_csv: Optional[Union[str, Path]] = None,
    patient_ids: Optional[Sequence[str]] = None,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    is_training: bool = False,
    roi_size: Sequence[int] = (128, 128, 64),
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    use_cache: bool = False,
    cache_rate: float = 1.0,
) -> DataLoader:
    """Create a high-performance PyTorch/MONAI DataLoader for 3D DCE-MRI.

    Args:
        data_dir: Directory containing DCE and mask NIfTI files.
        metadata_csv: Optional path to metadata CSV.
        patient_ids: Optional list of patient IDs.
        batch_size: Mini-batch size for training or validation.
        num_workers: Number of subprocesses for data loading.
        shuffle: Whether to shuffle data every epoch.
        is_training: Whether training mode (with data augmentations) is active.
        roi_size: Spatial target crop size (H, W, D).
        pixdim: Target voxel spacing (default: (1.0, 1.0, 1.0)).
        use_cache: Whether to use MONAI CacheDataset in RAM.
        cache_rate: Fraction of dataset to cache when use_cache=True.

    Returns:
        DataLoader: PyTorch / MONAI DataLoader.
    """
    transforms = build_dce_transforms(
        pixdim=pixdim,
        roi_size=roi_size,
        is_training=is_training,
    )

    dataset = BreastDCEDataset(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        patient_ids=patient_ids,
        transform=transforms,
    )

    if use_cache:
        dataset = CacheDataset(
            data=dataset.samples,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
