"""Test and visual verification script for 3D DCE-MRI data loading and preprocessing.

Tests both dummy data and real BreastDCEDL I-SPY1 dataset, prints tensor shapes,
and generates multi-slice diagnostic plots.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocessing.dataset import BreastDCEDataset, build_dce_transforms, get_dce_dataloader
from src.utils.make_dummy_data import make_dummy_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TestPreprocessing")


def generate_slice_plot(
    sample: Dict[str, Any],
    output_png_path: Path,
    title_prefix: str = "3D DCE-MRI Preprocessing",
) -> Path:
    """Generate and save multi-panel slice visualization plot.

    Panels:
    1. Pre-contrast (T0) slice
    2. Post-contrast (T1) slice
    3. Subtraction (T1 - T0) slice
    4. Tumor Mask overlay
    5. Multi-channel DCE composite / Cropped ROI slices

    Args:
        sample: Dictionary containing loaded and preprocessed tensors.
        output_png_path: Destination path for saving plot.
        title_prefix: Title prefix for figure.

    Returns:
        Path: Path to saved figure.
    """
    image = sample["image"]  # (3, H, W, D) -> [T0, T1, Sub]
    t0 = sample["t0"]        # (1, H, W, D)
    t1 = sample["t1"]        # (1, H, W, D)
    sub = sample["subtraction"]  # (1, H, W, D)
    mask = sample.get("mask", None)  # (1, H, W, D)

    # Convert to numpy
    t0_np = t0[0].cpu().numpy()
    t1_np = t1[0].cpu().numpy()
    sub_np = sub[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy() if mask is not None else np.zeros_like(t0_np)

    # Find the axial slice with maximum tumor mask area, or middle slice
    if mask_np.sum() > 0:
        slice_sums = mask_np.sum(axis=(0, 1))  # sum over (H, W) per depth slice
        slice_idx = int(np.argmax(slice_sums))
    else:
        slice_idx = t0_np.shape[2] // 2

    # Extract 2D axial slices
    t0_slice = t0_np[:, :, slice_idx]
    t1_slice = t1_np[:, :, slice_idx]
    sub_slice = sub_np[:, :, slice_idx]
    mask_slice = mask_np[:, :, slice_idx]

    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")

    # Style helper
    def style_ax(ax, title: str):
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # 1. Pre-contrast T0
    im0 = axes[0].imshow(t0_slice.T, cmap="gray", origin="lower")
    style_ax(axes[0], f"Pre-contrast (T0)\nSlice {slice_idx}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Post-contrast T1
    im1 = axes[1].imshow(t1_slice.T, cmap="gray", origin="lower")
    style_ax(axes[1], f"Post-contrast (T1)\nSlice {slice_idx}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Subtraction (T1 - T0)
    im2 = axes[2].imshow(sub_slice.T, cmap="magma", origin="lower")
    style_ax(axes[2], f"Subtraction (T1 - T0)\nSlice {slice_idx}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Mask Overlay on Subtraction
    axes[3].imshow(sub_slice.T, cmap="gray", origin="lower")
    if mask_slice.sum() > 0:
        masked_overlay = np.ma.masked_where(mask_slice.T <= 0.5, mask_slice.T)
        axes[3].imshow(masked_overlay, cmap="autumn", alpha=0.6, origin="lower")
    style_ax(axes[3], f"Tumor Mask Overlay\nArea: {int(mask_slice.sum())} voxels")

    # 5. Multi-channel composite (RGB from Sub, T1, T0)
    # Normalize 0-1 for RGB display
    def minmax(a):
        vmin, vmax = a.min(), a.max()
        return (a - vmin) / (vmax - vmin + 1e-7)

    rgb = np.stack([
        minmax(sub_slice.T),  # Red = Subtraction enhancement
        minmax(t1_slice.T),   # Green = Post-contrast
        minmax(t0_slice.T),   # Blue = Pre-contrast
    ], axis=-1)
    axes[4].imshow(rgb, origin="lower")
    style_ax(axes[4], "RGB Multiparametric\n[R:Sub, G:T1, B:T0]")

    pid = sample.get("pid", "Unknown")
    plt.suptitle(
        f"{title_prefix} - Subject: {pid} | Cropped Shape: {list(image.shape)}",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved multi-panel slice visualization to %s", output_png_path)
    return output_png_path


def run_tests() -> None:
    """Run data loading and preprocessing verification on dummy and real datasets."""
    logger.info("=== Starting 3D DCE-MRI Preprocessing Verification ===")

    # Step 1: Generate dummy synthetic dataset
    dummy_dir = Path("data/dummy")
    make_dummy_dataset(dummy_dir, num_subjects=3, shape=(60, 256, 256))

    # Step 2: Load dummy dataset through DataLoader
    logger.info("Testing DataLoader with dummy dataset...")
    dummy_loader = get_dce_dataloader(
        data_dir=dummy_dir,
        metadata_csv=dummy_dir / "BreastDCEDL_spy1_metadata.csv",
        batch_size=2,
        shuffle=False,
        roi_size=(128, 128, 64),
        pixdim=(1.0, 1.0, 1.0),
    )

    batch = next(iter(dummy_loader))
    logger.info("Dummy Batch Loaded Successfully:")
    logger.info("  - image (multi-channel DCE): %s (dtype: %s)", tuple(batch["image"].shape), batch["image"].dtype)
    logger.info("  - t0 (pre-contrast):         %s (min=%.2f, max=%.2f)", tuple(batch["t0"].shape), batch["t0"].min(), batch["t0"].max())
    logger.info("  - t1 (post-contrast):        %s (min=%.2f, max=%.2f)", tuple(batch["t1"].shape), batch["t1"].min(), batch["t1"].max())
    logger.info("  - subtraction (T1 - T0):     %s (min=%.2f, max=%.2f)", tuple(batch["subtraction"].shape), batch["subtraction"].min(), batch["subtraction"].max())
    logger.info("  - mask (tumor segmentation): %s (unique: %s)", tuple(batch["mask"].shape), torch.unique(batch["mask"]).tolist())
    logger.info("  - label (pCR target):        %s", batch["label"].tolist())
    logger.info("  - pids:                      %s", batch["pid"])

    # Step 3: Test on real BreastDCEDL_spy1 dataset if available
    real_data_dir = Path("BreastDCEDL_spy1")
    target_sample = None
    if real_data_dir.exists():
        logger.info("\nTesting DataLoader on actual BreastDCEDL_spy1 dataset...")
        real_dataset = BreastDCEDataset(
            data_dir=real_data_dir,
            metadata_csv=real_data_dir / "BreastDCEDL_spy1_metadata.csv",
            transform=build_dce_transforms(roi_size=(128, 128, 64), pixdim=(1.0, 1.0, 1.0)),
        )
        sample = real_dataset[0]
        target_sample = sample
        logger.info("Real Dataset Sample 0 Loaded:")
        logger.info("  - Patient ID: %s", sample["pid"])
        logger.info("  - image shape: %s", tuple(sample["image"].shape))
        logger.info("  - subtraction shape: %s", tuple(sample["subtraction"].shape))
        logger.info("  - mask shape: %s (sum: %d voxels)", tuple(sample["mask"].shape), int(sample["mask"].sum()))
        logger.info("  - label: %s", sample["label"])
        logger.info("  - metadata: %s", sample.get("metadata", "N/A"))
    else:
        # Fallback to dummy sample for plotting
        target_sample = {k: v[0] for k, v in batch.items()}

    # Step 4: Generate visualization artifacts
    artifact_dir = Path("reports/figures")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_path = artifact_dir / "sample_dce_slices.png"
    generate_slice_plot(
        target_sample,
        plot_path,
        title_prefix="BreastDCEDL I-SPY1 3D DCE-MRI",
    )

    logger.info("\n=== Preprocessing Verification PASSED! ===")


if __name__ == "__main__":
    run_tests()
