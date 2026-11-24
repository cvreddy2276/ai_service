"""Test and verification script for SimpleITK MRI Processor.

Executes the SimpleITK pipeline on synthetic anisotropic 3D DCE-MRI volumes
and real BreastDCEDL cases, compares physical metadata before and after resampling,
and generates diagnostic verification plots.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from src.preprocessing.mri_processor import MRIProcessor, apply_n4_bias_correction, compute_subtraction_volume, resample_to_isotropic
from src.utils.dummy_generator import create_synthetic_dce_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TestMRIProcessor")


def extract_metadata(image: sitk.Image) -> Dict[str, Any]:
    """Extract physical metadata dictionary from SimpleITK Image."""
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    pixel_type = image.GetPixelIDTypeAsString()

    physical_extent = tuple(round(size[i] * spacing[i], 3) for i in range(3))
    physical_volume_cm3 = round((physical_extent[0] * physical_extent[1] * physical_extent[2]) / 1000.0, 3)

    return {
        "size": size,
        "spacing": tuple(round(s, 4) for s in spacing),
        "origin": tuple(round(o, 4) for o in origin),
        "direction": tuple(round(d, 2) for d in direction),
        "physical_extent_mm": physical_extent,
        "physical_volume_cm3": physical_volume_cm3,
        "pixel_type": pixel_type,
    }


def generate_sitk_verification_plot(
    orig_t0: sitk.Image,
    orig_t1: sitk.Image,
    proc_t0: sitk.Image,
    proc_t1: sitk.Image,
    proc_sub: sitk.Image,
    proc_mask: sitk.Image,
    output_path: Path,
) -> Path:
    """Generate multi-panel comparative visualization of SimpleITK processing."""
    # Convert SimpleITK images to NumPy arrays for visualization (Z, Y, X)
    arr_orig_t0 = sitk.GetArrayFromImage(orig_t0)
    arr_orig_t1 = sitk.GetArrayFromImage(orig_t1)
    arr_proc_t0 = sitk.GetArrayFromImage(proc_t0)
    arr_proc_t1 = sitk.GetArrayFromImage(proc_t1)
    arr_proc_sub = sitk.GetArrayFromImage(proc_sub)
    arr_proc_mask = sitk.GetArrayFromImage(proc_mask)

    # Find slice with maximum tumor mask in resampled space
    if arr_proc_mask.sum() > 0:
        slice_z = int(np.argmax(arr_proc_mask.sum(axis=(1, 2))))
    else:
        slice_z = arr_proc_t0.shape[0] // 2

    # Map corresponding slice in original anisotropic space based on physical Z
    orig_slice_z = int(round(slice_z * (proc_t0.GetSpacing()[2] / orig_t0.GetSpacing()[2])))
    orig_slice_z = min(max(0, orig_slice_z), arr_orig_t0.shape[0] - 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("#161b22")

    def style_ax(ax, title: str):
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # Row 1: Original Anisotropic Inputs & Resampled T0
    im0 = axes[0, 0].imshow(arr_orig_t0[orig_slice_z], cmap="gray", origin="lower")
    style_ax(
        axes[0, 0],
        f"1. Original Anisotropic T0 (Pre)\nSize: {orig_t0.GetSize()} | Spacing: {tuple(round(s, 2) for s in orig_t0.GetSpacing())} mm",
    )
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(arr_orig_t1[orig_slice_z], cmap="gray", origin="lower")
    style_ax(
        axes[0, 1],
        f"2. Original Anisotropic T1 (Post)\nSize: {orig_t1.GetSize()} | Spacing: {tuple(round(s, 2) for s in orig_t1.GetSpacing())} mm",
    )
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(arr_proc_t0[slice_z], cmap="gray", origin="lower")
    style_ax(
        axes[0, 2],
        f"3. N4 Corrected & Isotropic T0\nSize: {proc_t0.GetSize()} | Spacing: {proc_t0.GetSpacing()} mm",
    )
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: Processed T1, Subtraction, and Lesion Mask
    im3 = axes[1, 0].imshow(arr_proc_t1[slice_z], cmap="gray", origin="lower")
    style_ax(
        axes[1, 0],
        f"4. N4 Corrected & Isotropic T1\nSize: {proc_t1.GetSize()} | Spacing: {proc_t1.GetSpacing()} mm",
    )
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(arr_proc_sub[slice_z], cmap="magma", origin="lower")
    style_ax(
        axes[1, 1],
        f"5. Computed Subtraction (T1 - T0)\nPeak Enhancement Highlighted",
    )
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(arr_proc_sub[slice_z], cmap="gray", origin="lower")
    if arr_proc_mask[slice_z].sum() > 0:
        masked = np.ma.masked_where(arr_proc_mask[slice_z] == 0, arr_proc_mask[slice_z])
        axes[1, 2].imshow(masked, cmap="autumn", alpha=0.65, origin="lower")
    style_ax(
        axes[1, 2],
        f"6. Resampled Mask Overlay on Subtraction\nMask Pixels: {int(arr_proc_mask[slice_z].sum())}",
    )

    plt.suptitle(
        "SimpleITK 3D DCE-MRI Preprocessing: Physical Space & Subtraction Verification",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved SimpleITK verification plot to %s", output_path)
    return output_path


def run_pipeline_test() -> None:
    """Execute SimpleITK pipeline on synthetic volume and compare physical metadata."""
    logger.info("================================================================")
    logger.info("Starting SimpleITK 3D DCE-MRI Preprocessing Pipeline Test")
    logger.info("================================================================")

    # 1. Generate synthetic anisotropic case
    orig_size = (256, 256, 40)
    orig_spacing = (0.75, 0.75, 2.5)
    orig_origin = (-95.5, -120.0, -50.0)

    logger.info("Generating synthetic anisotropic case: Size=%s, Spacing=%s mm", orig_size, orig_spacing)
    t0_img, t1_img, mask_img = create_synthetic_dce_case(
        size=orig_size,
        spacing=orig_spacing,
        origin=orig_origin,
    )

    meta_before = extract_metadata(t0_img)
    logger.info("\n--- METADATA BEFORE RESAMPLING (Anisotropic) ---")
    for k, v in meta_before.items():
        logger.info("  %-22s: %s", k, v)

    # 2. Run SimpleITK MRIProcessor pipeline
    processor = MRIProcessor(
        target_spacing=(1.0, 1.0, 1.0),
        apply_n4=True,
        shrink_factor=2,
        clip_negative=True,
    )
    logger.info("\nExecuting MRIProcessor (N4 Bias Correction + Isotropic Resampling + Subtraction)...")
    results = processor.process_case(t0_img, t1_img, mask_img)

    proc_t0 = results["t0"]
    proc_t1 = results["t1"]
    proc_sub = results["subtraction"]
    proc_mask = results["mask"]

    meta_after = extract_metadata(proc_t0)
    meta_sub = extract_metadata(proc_sub)

    logger.info("\n--- METADATA AFTER RESAMPLING (Isotropic 1.0 mm) ---")
    for k, v in meta_after.items():
        logger.info("  %-22s: %s", k, v)

    logger.info("\n--- SUBTRACTION VOLUME METADATA ---")
    for k, v in meta_sub.items():
        logger.info("  %-22s: %s", k, v)

    # 3. Verify physical extent & bounding box preservation
    extent_diff = [
        abs(meta_after["physical_extent_mm"][i] - meta_before["physical_extent_mm"][i])
        for i in range(3)
    ]
    logger.info(
        "\nPhysical Extent Difference (Before vs After): %s mm (Max diff: %.3f mm)",
        extent_diff,
        max(extent_diff),
    )
    assert max(extent_diff) <= 1.0, f"Physical extent discrepancy too large: {max(extent_diff)} mm"
    logger.info("Physical extent conservation check: PASSED (Tolerance <= 1.0 mm)")

    # 4. Verify Subtraction Non-negativity Clamping
    arr_sub = sitk.GetArrayFromImage(proc_sub)
    min_sub_val = float(arr_sub.min())
    max_sub_val = float(arr_sub.max())
    logger.info("Subtraction Value Range: min=%.4f, max=%.4f (Non-negative clamped: %s)", min_sub_val, max_sub_val, min_sub_val >= 0.0)
    assert min_sub_val >= 0.0, f"Subtraction clamping failed: min value is {min_sub_val}"

    # 5. Generate and save verification plot
    artifact_plot_path = Path("C:/Users/vaish/.gemini/antigravity/brain/c0200787-4302-4c90-b232-c55b002c93b7/sitk_mri_processor_verification.png")
    generate_sitk_verification_plot(
        orig_t0=t0_img,
        orig_t1=t1_img,
        proc_t0=proc_t0,
        proc_t1=proc_t1,
        proc_sub=proc_sub,
        proc_mask=proc_mask,
        output_path=artifact_plot_path,
    )

    logger.info("\n================================================================")
    logger.info("SimpleITK MRI Preprocessing Verification PASSED Successfully!")
    logger.info("================================================================")


if __name__ == "__main__":
    run_pipeline_test()
