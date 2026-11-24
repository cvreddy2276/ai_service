"""Synthetic 3D DCE-MRI Volume Generator using SimpleITK.

Generates mock 3D NIfTI volumes with anisotropic physical spacing, non-zero origin,
intensity bias field, and binary lesion masks for offline pipeline testing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SimpleITKDummyGenerator")


def create_synthetic_dce_case(
    size: Tuple[int, int, int] = (256, 256, 40),
    spacing: Tuple[float, float, float] = (0.75, 0.75, 2.5),
    origin: Tuple[float, float, float] = (-95.5, -120.0, -50.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    tumor_radius_voxels: Tuple[int, int, int] = (14, 14, 5),
    tumor_center_offset: Tuple[int, int, int] = (20, -15, 0),
    add_bias_field: bool = True,
    noise_sigma: float = 3.0,
    seed: int = 42,
) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    """Generate synthetic 3D pre-contrast, post-contrast, and lesion mask SimpleITK images.

    Args:
        size: Volume dimensions (X, Y, Z) in voxels (default: (256, 256, 40)).
        spacing: Physical voxel spacing (dx, dy, dz) in mm (default: (0.75, 0.75, 2.5)).
        origin: Physical origin (ox, oy, oz) in mm (default: (-95.5, -120.0, -50.0)).
        direction: Direction cosine matrix (default: identity).
        tumor_radius_voxels: Ellipsoid radius (rx, ry, rz) in voxels.
        tumor_center_offset: Lesion offset (dx, dy, dz) from volume center.
        add_bias_field: Whether to inject low-frequency spatial intensity bias.
        noise_sigma: Gaussian noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple[sitk.Image, sitk.Image, sitk.Image]: (T0 Image, T1 Image, Binary Mask Image).
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = size

    # SimpleITK image size (X, Y, Z) corresponds to NumPy array shape (Z, Y, X)
    # Coordinate grid:
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")

    # 1. Breast parenchymal tissue contour (3D hemi-ellipsoid)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    rx, ry, rz = nx // 2 - 15, ny // 2 - 15, nz // 2 - 2

    dist_breast = (
        ((xx - cx) / max(1, rx)) ** 2
        + ((yy - cy) / max(1, ry)) ** 2
        + ((zz - cz) / max(1, rz)) ** 2
    )
    breast_mask = dist_breast <= 1.0

    # 2. Fibroglandular base tissue signal
    base_tissue = np.zeros((nz, ny, nx), dtype=np.float32)
    base_tissue[breast_mask] = (
        120.0
        + 25.0 * np.sin(xx[breast_mask] / 12.0)
        + 15.0 * np.cos(yy[breast_mask] / 10.0)
    )

    # 3. Low-frequency spatial bias field modulation (simulating B1 inhomogeneity)
    if add_bias_field:
        bias_field = 1.0 + 0.35 * np.exp(-(((xx - cx * 0.4) / (nx * 0.6)) ** 2 + ((yy - cy * 0.4) / (ny * 0.6)) ** 2))
        base_tissue *= bias_field.astype(np.float32)

    # 4. Focal 3D hypervascular lesion (tumor)
    lx = cx + tumor_center_offset[0]
    ly = cy + tumor_center_offset[1]
    lz = cz + tumor_center_offset[2]

    dist_tumor = (
        ((xx - lx) / max(1, tumor_radius_voxels[0])) ** 2
        + ((yy - ly) / max(1, tumor_radius_voxels[1])) ** 2
        + ((zz - lz) / max(1, tumor_radius_voxels[2])) ** 2
    )
    tumor_mask = (dist_tumor <= 1.0).astype(np.uint8)

    # 5. Construct T0 pre-contrast volume
    t0_arr = base_tissue.copy()
    t0_arr[tumor_mask > 0] = 115.0 + rng.normal(0, noise_sigma, size=int(tumor_mask.sum()))
    t0_arr += rng.normal(0, noise_sigma, size=t0_arr.shape)
    t0_arr[~breast_mask] = np.clip(rng.normal(6.0, 2.0, size=(~breast_mask).sum()), 0.0, None)
    t0_arr = np.clip(t0_arr, 0.0, None).astype(np.float32)

    # 6. Construct T1 post-contrast volume (marked focal enhancement in lesion)
    t1_arr = base_tissue.copy() * 1.06  # mild parenchymal enhancement
    t1_arr[tumor_mask > 0] = 320.0 + rng.normal(0, noise_sigma * 2, size=int(tumor_mask.sum()))  # ~3x uptake
    t1_arr += rng.normal(0, noise_sigma, size=t1_arr.shape)
    t1_arr[~breast_mask] = np.clip(rng.normal(6.0, 2.0, size=(~breast_mask).sum()), 0.0, None)
    t1_arr = np.clip(t1_arr, 0.0, None).astype(np.float32)

    # 7. Convert NumPy arrays to SimpleITK Images and assign physical geometry
    t0_img = sitk.GetImageFromArray(t0_arr)
    t1_img = sitk.GetImageFromArray(t1_arr)
    mask_img = sitk.GetImageFromArray(tumor_mask)

    for img in (t0_img, t1_img, mask_img):
        img.SetSpacing(list(spacing))
        img.SetOrigin(list(origin))
        img.SetDirection(list(direction))

    logger.debug(
        "Created synthetic SimpleITK case: size=%s, spacing=%s, origin=%s",
        t0_img.GetSize(),
        t0_img.GetSpacing(),
        t0_img.GetOrigin(),
    )
    return t0_img, t1_img, mask_img


def save_synthetic_dce_case(
    output_dir: Union[str, Path],
    case_id: str = "ISPY1_synthetic001",
    size: Tuple[int, int, int] = (256, 256, 40),
    spacing: Tuple[float, float, float] = (0.75, 0.75, 2.5),
) -> Tuple[Path, Path, Path]:
    """Generate and save synthetic DCE-MRI NIfTI files to disk.

    Args:
        output_dir: Destination directory.
        case_id: Identifier prefix for generated files.
        size: Image size (X, Y, Z).
        spacing: Voxel spacing (dx, dy, dz) in mm.

    Returns:
        Tuple[Path, Path, Path]: Paths to saved (T0, T1, Mask) NIfTI files.
    """
    out_dir = Path(output_dir)
    dce_dir = out_dir / "spt1_dce"
    mask_dir = out_dir / "spy1_mask"
    dce_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    t0_img, t1_img, mask_img = create_synthetic_dce_case(size=size, spacing=spacing)

    t0_path = dce_dir / f"{case_id}_spy1_vis1_acq0.nii.gz"
    t1_path = dce_dir / f"{case_id}_spy1_vis1_acq1.nii.gz"
    mask_path = mask_dir / f"{case_id}_spy1_vis1_mask.nii.gz"

    sitk.WriteImage(t0_img, str(t0_path))
    sitk.WriteImage(t1_img, str(t1_path))
    sitk.WriteImage(mask_img, str(mask_path))

    logger.info("Saved synthetic SimpleITK case %s to %s", case_id, out_dir)
    return t0_path, t1_path, mask_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SimpleITK 3D DCE-MRI volumes.")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_sitk", help="Output directory")
    parser.add_argument("--case_id", type=str, default="ISPY1_synthetic001", help="Case identifier")
    args = parser.parse_args()

    t0_p, t1_p, mask_p = save_synthetic_dce_case(args.output_dir, case_id=args.case_id)
    print(f"Generated synthetic case files:\n  T0:   {t0_p}\n  T1:   {t1_p}\n  Mask: {mask_p}")
