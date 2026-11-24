"""Medical Imaging Preprocessing & Visualization Utilities for DCE-MRI."""

from __future__ import annotations

import base64
import io
import logging
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_nifti_from_bytes(
    file_bytes: bytes,
    filename: str = "scan.nii.gz",
) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
    """Load 3D NIfTI volume from in-memory byte buffer.

    Args:
        file_bytes: Raw bytes of the uploaded .nii or .nii.gz file.
        filename: Original file name.

    Returns:
        Tuple containing:
            - Canonical RAS ndarray of shape (X, Y, Z), resampled to 1 mm spacing.
            - Result voxel spacing (1.0, 1.0, 1.0) mm.
            - Updated 4x4 affine matrix for the resampled volume.

    Raises:
        ValueError: If file cannot be parsed as a valid NIfTI object.
    """
    try:
        if filename.endswith(".gz"):
            # Gzip compressed NIfTI
            import gzip
            decompressed = gzip.decompress(file_bytes)
            nii_img = nib.Nifti1Image.from_bytes(decompressed)
        else:
            nii_img = nib.Nifti1Image.from_bytes(file_bytes)

        nii_img = nib.as_closest_canonical(nii_img)
        data = nii_img.get_fdata(dtype=np.float32)
        zooms = tuple(float(z) for z in nii_img.header.get_zooms()[:3])
        affine = nii_img.affine

        # If 4D DCE-MRI (X, Y, Z, T), subtract the first post-contrast phase from pre-contrast.
        if data.ndim == 4:
            logger.info("4D DCE-MRI detected with shape %s; computing phase-1 minus phase-0 subtraction.", data.shape)
            if data.shape[-1] >= 2:
                # Subtraction I_sub = I_post - I_pre
                data = np.clip(data[..., 1] - data[..., 0], 0, None)
            else:
                raise ValueError("4D DCE input must contain at least a pre-contrast and post-contrast phase.")

        resampled = resample_to_output(nib.Nifti1Image(data, affine), voxel_sizes=(1.0, 1.0, 1.0), order=1)
        data = resampled.get_fdata(dtype=np.float32)
        return data, (1.0, 1.0, 1.0), resampled.affine
    except Exception as e:
        logger.error("Failed to parse NIfTI volume from bytes: %s", e, exc_info=True)
        raise ValueError(f"Corrupted or invalid NIfTI file: {e}") from e


def load_nifti_from_path(
    file_path: str,
) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
    """Load 3D NIfTI volume from disk file path.

    Args:
        file_path: Filesystem path to .nii or .nii.gz file.

    Returns:
        Tuple containing (volume_array, spacing_zooms, affine_matrix).
    """
    nii_img = nib.as_closest_canonical(nib.load(file_path))
    data = nii_img.get_fdata(dtype=np.float32)
    zooms = tuple(float(z) for z in nii_img.header.get_zooms()[:3])
    affine = nii_img.affine

    if data.ndim == 4:
        if data.shape[-1] >= 2:
            data = np.clip(data[..., 1] - data[..., 0], 0, None)
        else:
            raise ValueError("4D DCE input must contain at least a pre-contrast and post-contrast phase.")

    resampled = resample_to_output(nib.Nifti1Image(data, affine), voxel_sizes=(1.0, 1.0, 1.0), order=1)
    data = resampled.get_fdata(dtype=np.float32)
    return data, (1.0, 1.0, 1.0), resampled.affine


def normalize_volume(
    volume: np.ndarray,
) -> np.ndarray:
    """Z-score normalize non-zero voxels, matching the training transform.

    Args:
        volume: Raw 3D volume array.
    Returns:
        Z-score normalized volume with background voxels set to zero.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume after temporal handling; got shape {volume.shape}.")
    mask = np.isfinite(volume) & (volume != 0)
    if not np.any(mask):
        raise ValueError("Volume has no finite non-zero voxels to normalize.")
    values = volume[mask]
    mean = float(values.mean())
    std = float(values.std())
    normalized = np.zeros_like(volume, dtype=np.float32)
    if std > 1e-7:
        normalized[mask] = (values - mean) / (std + 1e-7)
    return normalized


def resize_volume_to_tensor(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (128, 128, 64),
) -> torch.Tensor:
    """Center-crop or zero-pad a 3D volume to (X,Y,Z) and return (B=1,C=1,X,Y,Z).

    Args:
        volume: 3D numpy array of shape (X, Y, Z).
        target_shape: Expected canonical (X, Y, Z) shape.

    Returns:
        PyTorch tensor of shape (1, 1, X, Y, Z).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    result = np.zeros(target_shape, dtype=np.float32)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(volume.shape, target_shape):
        length = min(source_size, target_size)
        source_start = (source_size - length) // 2
        target_start = (target_size - length) // 2
        source_slices.append(slice(source_start, source_start + length))
        target_slices.append(slice(target_start, target_start + length))
    result[tuple(target_slices)] = volume[tuple(source_slices)]
    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0)


def slice_to_base64_png(
    slice_2d: np.ndarray,
    heatmap_2d: Optional[np.ndarray] = None,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> str:
    """Render a 2D MRI slice with optional Grad-CAM heatmap overlay as a base64 encoded PNG.

    Args:
        slice_2d: 2D array of normalized pixel intensities [0, 1].
        heatmap_2d: Optional 2D Grad-CAM activation map [0, 1].
        alpha: Blending opacity for the heatmap.
        colormap: Matplotlib colormap name ('jet', 'turbo', 'inferno').

    Returns:
        Base64 data URI string "data:image/png;base64,...".
    """
    # Ensure slice is 2D
    slice_norm = np.clip(slice_2d, 0.0, 1.0)
    # Rotate 90 deg counter-clockwise for standard radiologic orientation
    slice_norm = np.rot90(slice_norm)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Base grayscale anatomical MRI
    ax.imshow(slice_norm, cmap="gray", vmin=0.0, vmax=1.0)

    # Optional Grad-CAM color heatmap overlay
    if heatmap_2d is not None:
        heat = np.rot90(heatmap_2d)
        heat = np.clip(heat, 0.0, 1.0)
        # Threshold low activation noise to transparent
        heat_masked = np.ma.masked_where(heat < 0.15, heat)
        cmap = plt.get_cmap(colormap)
        ax.imshow(heat_masked, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
