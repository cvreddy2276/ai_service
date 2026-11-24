"""Synthetic 3D DCE-MRI Breast Cancer dataset generator for testing and CI/CD.

Generates realistic mock NIfTI volumes (T0 pre-contrast, T1 post-contrast with
focal tumor enhancement, and 3D binary ground-truth segmentation masks)
matching the I-SPY1 BreastDCEDL format.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MakeDummyData")


def generate_synthetic_subject(
    shape: Tuple[int, int, int] = (64, 128, 128),
    spacing: Tuple[float, float, float] = (1.5, 0.8, 0.8),
    tumor_radius: Tuple[int, int, int] = (6, 12, 12),
    tumor_center_offset: Tuple[int, int, int] = (0, 10, -15),
    noise_level: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D pre-contrast, post-contrast, and tumor mask arrays.

    Args:
        shape: 3D spatial dimensions (Depth, Height, Width).
        spacing: Voxel spacing (dz, dy, dx) in mm.
        tumor_radius: Ellipsoid semi-axes (rz, ry, rx) in voxels.
        tumor_center_offset: Shift of tumor center relative to volume center.
        noise_level: Gaussian noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (T0 array, T1 array, Mask array).
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape

    # Coordinate grids
    z = np.arange(D)
    y = np.arange(H)
    x = np.arange(W)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    # 1. Breast parenchymal background (ellipsoid breast region)
    breast_cz, breast_cy, breast_cx = D // 2, H // 2, W // 2
    breast_rz, breast_ry, breast_rx = D // 2 - 4, H // 2 - 10, W // 2 - 10

    breast_dist = (
        ((zz - breast_cz) / max(1, breast_rz)) ** 2
        + ((yy - breast_cy) / max(1, breast_ry)) ** 2
        + ((xx - breast_cx) / max(1, breast_rx)) ** 2
    )
    breast_mask = breast_dist <= 1.0

    # Fibroglandular base tissue signal
    base_signal = np.zeros(shape, dtype=np.float32)
    base_signal[breast_mask] = 100.0 + 30.0 * np.sin(yy[breast_mask] / 8.0) + 20.0 * np.cos(xx[breast_mask] / 10.0)

    # 2. Focal 3D tumor lesion (ellipsoid with high post-contrast wash-in)
    tc_z = breast_cz + tumor_center_offset[0]
    tc_y = breast_cy + tumor_center_offset[1]
    tc_x = breast_cx + tumor_center_offset[2]

    tumor_dist = (
        ((zz - tc_z) / max(1, tumor_radius[0])) ** 2
        + ((yy - tc_y) / max(1, tumor_radius[1])) ** 2
        + ((xx - tc_x) / max(1, tumor_radius[2])) ** 2
    )
    tumor_mask = (tumor_dist <= 1.0).astype(np.float32)

    # 3. Pre-contrast T0 volume: baseline tissue + mild tumor hypointensity / isointensity
    t0 = base_signal.copy()
    t0[tumor_mask > 0] = 95.0 + rng.normal(0, 5.0, size=int(tumor_mask.sum()))
    t0 += rng.normal(0, noise_level * 100.0, size=shape)
    t0 = np.clip(t0, 0.0, None)
    t0[~breast_mask] = np.clip(rng.normal(5.0, 2.0, size=(~breast_mask).sum()), 0.0, None)

    # 4. Post-contrast T1 volume: strong hyperintense enhancement inside tumor (contrast uptake)
    t1 = base_signal.copy() * 1.05  # slight background parenchyma enhancement (BPE)
    # Strong dynamic uptake in tumor (2.5x - 3.0x signal increase)
    t1[tumor_mask > 0] = 280.0 + rng.normal(0, 15.0, size=int(tumor_mask.sum()))
    t1 += rng.normal(0, noise_level * 100.0, size=shape)
    t1 = np.clip(t1, 0.0, None)
    t1[~breast_mask] = np.clip(rng.normal(5.0, 2.0, size=(~breast_mask).sum()), 0.0, None)

    return t0.astype(np.float32), t1.astype(np.float32), tumor_mask.astype(np.float32)


def make_dummy_dataset(
    output_dir: Path,
    num_subjects: int = 3,
    shape: Tuple[int, int, int] = (60, 256, 256),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Generate and save mock NIfTI files and clinical metadata CSV.

    Args:
        output_dir: Destination directory for mock data.
        num_subjects: Number of synthetic subjects to create.
        shape: 3D volume dimensions (Depth, Height, Width).
        spacing: Voxel resolution (dz, dy, dx) in mm.

    Returns:
        Path: Path to created output directory.
    """
    output_dir = Path(output_dir)
    dce_dir = output_dir / "spt1_dce"
    mask_dir = output_dir / "spy1_mask"
    dce_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    logger.info("Generating %d synthetic 3D DCE-MRI subjects in %s...", num_subjects, output_dir)

    for i in range(1, num_subjects + 1):
        pid = f"ISPY1_dummy{i:03d}"
        seed = 100 + i

        # Vary tumor size and location per dummy subject
        radius = (5 + i % 3, 10 + (i * 2) % 6, 10 + (i * 3) % 6)
        offset = ((i % 3 - 1) * 4, 15 - (i * 8), -15 + (i * 10))
        t0, t1, mask = generate_synthetic_subject(
            shape=shape,
            spacing=spacing,
            tumor_radius=radius,
            tumor_center_offset=offset,
            seed=seed,
        )

        # Save NIfTIs
        t0_nii = nib.Nifti1Image(t0, affine)
        t0_nii.header.set_zooms(spacing)
        t0_path = dce_dir / f"{pid}_spy1_vis1_acq0.nii.gz"
        nib.save(t0_nii, str(t0_path))

        t1_nii = nib.Nifti1Image(t1, affine)
        t1_nii.header.set_zooms(spacing)
        t1_path = dce_dir / f"{pid}_spy1_vis1_acq1.nii.gz"
        nib.save(t1_nii, str(t1_path))

        mask_nii = nib.Nifti1Image(mask, affine)
        mask_nii.header.set_zooms(spacing)
        mask_path = mask_dir / f"{pid}_spy1_vis1_mask.nii.gz"
        nib.save(mask_nii, str(mask_path))

        # Clinical metadata record
        pcr = 1.0 if i % 2 == 0 else 0.0
        metadata_records.append({
            "pid": pid,
            "age": 40.0 + i * 5.2,
            "ER": 1.0 if i % 2 == 1 else 0.0,
            "PR": 1.0 if i % 2 == 1 else 0.0,
            "HER2": 0.0 if i % 3 != 0 else 1.0,
            "HR_HER2_STATUS": "HRposHER2neg" if i % 2 == 1 else "TripleNeg",
            "pCR": pcr,
            "rcb_class": 0.0 if pcr == 1.0 else 2.0,
        })
        logger.info("  Created %s: T0, T1, and Mask saved. Volume: %s", pid, shape)

    # Save metadata CSV
    meta_df = pd.DataFrame(metadata_records)
    meta_csv_path = output_dir / "BreastDCEDL_spy1_metadata.csv"
    meta_df.to_csv(meta_csv_path, index=False)
    logger.info("Metadata CSV saved to %s", meta_csv_path)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 3D DCE-MRI dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dummy",
        help="Target folder for dummy dataset",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=3,
        help="Number of mock subjects to generate",
    )
    args = parser.parse_args()

    out_path = make_dummy_dataset(Path(args.output_dir), num_subjects=args.num_subjects)
    print(f"\nDummy dataset successfully generated at: {out_path.resolve()}")
