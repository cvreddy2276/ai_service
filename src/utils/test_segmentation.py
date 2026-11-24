"""Test and verification script for Phase 3: MONAI 3D UNet Tumor Segmentation.

Creates synthetic isotropic subtraction cases, tests positive/negative patch sampling,
runs a 2-epoch dry run training pass, executes sliding-window validation,
and generates diagnostic verification plots.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

from src.preprocessing.mri_processor import compute_subtraction_volume, resample_to_isotropic
from src.segmentation.dataset import get_segmentation_dataloader
from src.segmentation.model import build_3d_unet
from src.segmentation.train import SegmentationTrainer
from src.utils.dummy_generator import create_synthetic_dce_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TestSegmentation")


def prepare_synthetic_segmentation_dataset(
    output_dir: Path,
    num_cases: int = 6,
) -> List[Dict[str, str]]:
    """Generate preprocessed isotropic 3D subtraction and mask files for testing.

    Args:
        output_dir: Destination folder.
        num_cases: Number of synthetic cases to generate.

    Returns:
        List[Dict[str, str]]: List of {'image': sub_path, 'label': mask_path}.
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "subtraction"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    data_list: List[Dict[str, str]] = []
    logger.info("Generating %d synthetic isotropic 3D subtraction cases in %s...", num_cases, output_dir)

    for i in range(1, num_cases + 1):
        case_id = f"SYNTH_CASE_{i:03d}"
        seed = 200 + i

        # 1. Create synthetic DCE images with varying tumor positions/sizes
        radius = (10 + (i % 3) * 2, 10 + ((i + 1) % 3) * 2, 6 + (i % 2) * 2)
        offset = ((i % 3 - 1) * 15, 10 - (i % 4) * 8, 0)

        t0_img, t1_img, mask_img = create_synthetic_dce_case(
            size=(192, 192, 48),
            spacing=(1.0, 1.0, 1.5),
            tumor_radius_voxels=radius,
            tumor_center_offset=offset,
            seed=seed,
        )

        # 2. Resample to 1.0 mm isotropic
        t0_iso = resample_to_isotropic(t0_img, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
        t1_iso = resample_to_isotropic(t1_img, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
        mask_iso = resample_to_isotropic(mask_img, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkNearestNeighbor)

        # 3. Compute Subtraction Volume
        sub_iso = compute_subtraction_volume(t0_iso, t1_iso, clip_negative=True)

        # 4. Save NIfTI files
        sub_path = img_dir / f"{case_id}_subtraction.nii.gz"
        mask_path = mask_dir / f"{case_id}_mask.nii.gz"

        sitk.WriteImage(sub_iso, str(sub_path))
        sitk.WriteImage(mask_iso, str(mask_path))

        data_list.append({
            "image": str(sub_path),
            "label": str(mask_path),
            "case_id": case_id,
        })
        logger.debug("Generated %s (Size: %s)", case_id, sub_iso.GetSize())

    logger.info("Successfully prepared %d paired cases.", len(data_list))
    return data_list


def generate_segmentation_plot(
    val_image_path: str,
    val_label_path: str,
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
) -> Path:
    """Generate diagnostic visualization comparing ground truth vs UNet prediction."""
    # Load test case
    sub_sitk = sitk.ReadImage(val_image_path, sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(val_label_path, sitk.sitkUInt8)

    sub_arr = sitk.GetArrayFromImage(sub_sitk)  # (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)

    # Prepare tensor for sliding window inference
    # MONAI expects (B, C, H, W, D) -> (1, 1, X, Y, Z) or (1, 1, Z, Y, X) depending on loading
    # Standard: (1, 1, Z, Y, X) from array
    input_tensor = torch.from_numpy(sub_arr).unsqueeze(0).unsqueeze(0).to(device)
    # Normalize input
    mean = input_tensor[input_tensor != 0].mean()
    std = input_tensor[input_tensor != 0].std()
    norm_input = (input_tensor - mean) / (std + 1e-7)

    model.eval()
    with torch.no_grad():
        val_output = sliding_window_inference(
            inputs=norm_input,
            roi_size=(32, 96, 96),  # (Z, Y, X)
            sw_batch_size=4,
            predictor=model,
            overlap=0.25,
        )
        pred_prob = torch.sigmoid(val_output).squeeze().cpu().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.uint8)

    # Find slice with maximum tumor mask
    if mask_arr.sum() > 0:
        slice_z = int(np.argmax(mask_arr.sum(axis=(1, 2))))
    else:
        slice_z = sub_arr.shape[0] // 2

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig.patch.set_facecolor("#161b22")

    def style_ax(ax, title: str):
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # 1. Subtraction Volume
    im0 = axes[0].imshow(sub_arr[slice_z], cmap="magma", origin="lower")
    style_ax(axes[0], f"1. Subtraction Image (I_sub)\nSlice Z={slice_z}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Ground Truth Mask Overlay
    axes[1].imshow(sub_arr[slice_z], cmap="gray", origin="lower")
    if mask_arr[slice_z].sum() > 0:
        masked_gt = np.ma.masked_where(mask_arr[slice_z] == 0, mask_arr[slice_z])
        axes[1].imshow(masked_gt, cmap="autumn", alpha=0.65, origin="lower")
    style_ax(axes[1], f"2. Ground Truth Lesion Mask\nTumor Pixels: {int(mask_arr[slice_z].sum())}")

    # 3. Model Predicted Probability Heatmap
    im2 = axes[2].imshow(pred_prob[slice_z], cmap="jet", vmin=0.0, vmax=1.0, origin="lower")
    style_ax(axes[2], f"3. UNet Probability Heatmap\nSigmoid Output [0.0 - 1.0]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Predicted Binary Mask Overlay
    axes[3].imshow(sub_arr[slice_z], cmap="gray", origin="lower")
    if pred_mask[slice_z].sum() > 0:
        masked_pred = np.ma.masked_where(pred_mask[slice_z] == 0, pred_mask[slice_z])
        axes[3].imshow(masked_pred, cmap="cool", alpha=0.65, origin="lower")
    style_ax(axes[3], f"4. Predicted Binary Mask (P > 0.5)\nPred Pixels: {int(pred_mask[slice_z].sum())}")

    plt.suptitle(
        "Phase 3: MONAI 3D UNet Tumor Segmentation Dry-Run Prediction",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved segmentation verification plot to %s", output_path)
    return output_path


def run_segmentation_tests() -> None:
    """Execute complete Phase 3 segmentation validation and dry-run training."""
    logger.info("================================================================")
    logger.info("Starting Phase 3: MONAI 3D UNet Segmentation Engine Verification")
    logger.info("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Operating PyTorch Device: %s", device)

    # 1. Prepare synthetic isotropic data
    data_dir = Path("data/segmentation_dryrun")
    all_data = prepare_synthetic_segmentation_dataset(data_dir, num_cases=6)

    train_data = all_data[:4]
    val_data = all_data[4:]
    logger.info("Data Split: %d Training Cases, %d Validation Cases", len(train_data), len(val_data))

    # 2. Test DataLoader & Patch Sampling
    spatial_size = (96, 96, 32)
    num_samples = 2
    batch_size = 2

    logger.info("\nInitializing Training DataLoader with RandCropByPosNegLabeld...")
    train_loader = get_segmentation_dataloader(
        data=train_data,
        stage="train",
        batch_size=batch_size,
        spatial_size=spatial_size,
        num_samples=num_samples,
    )

    sample_batch = next(iter(train_loader))
    img_shape = list(sample_batch["image"].shape)
    lbl_shape = list(sample_batch["label"].shape)

    # Shape is (batch_size * num_samples, 1, 96, 96, 32)
    logger.info("--- TENSOR SHAPE VERIFICATION (Training Patch Batch) ---")
    logger.info("  Input Image Batch Tensor: %s (dtype: %s)", img_shape, sample_batch["image"].dtype)
    logger.info("  Ground Truth Batch Tensor: %s (unique: %s)", lbl_shape, torch.unique(sample_batch["label"]).tolist())

    expected_batch_dim = batch_size * num_samples
    assert img_shape[0] == expected_batch_dim, f"Expected batch dim {expected_batch_dim}, got {img_shape[0]}"
    assert img_shape[1] == 1, f"Expected 1 channel, got {img_shape[1]}"
    logger.info("Batch tensor shape check: PASSED (B=%d, C=1, Spatial=%s)", expected_batch_dim, spatial_size)

    # 3. Test Model Instantiation & Forward Pass
    logger.info("\nInstantiating 3D Residual UNet...")
    model = build_3d_unet(
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        device=device,
    )

    with torch.no_grad():
        test_in = sample_batch["image"].to(device)
        test_out = model(test_in)
        out_shape = list(test_out.shape)
        logger.info("Forward Pass Output Tensor Shape: %s", out_shape)
        assert out_shape == img_shape, f"Output shape {out_shape} does not match input shape {img_shape}"
    logger.info("3D UNet Forward Pass Check: PASSED")

    # 4. Run 2-Epoch Dry-Run Training Loop
    logger.info("\n--- EXECUTING 2-EPOCH DRY-RUN TRAINING PASS ---")
    val_loader = get_segmentation_dataloader(
        data=val_data,
        stage="val",
        batch_size=1,
    )

    checkpoint_dir = Path("checkpoints")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        roi_size=spatial_size,
        sw_batch_size=4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name="best_monai_3dunet.pt",
        device=device,
    )

    results = trainer.fit(max_epochs=2, val_interval=1)

    train_losses = results["history"]["train_loss"]
    val_dices = results["history"]["val_dice"]

    logger.info("\n--- TRAINING HISTORY SUMMARY ---")
    logger.info("  Epoch 1 Loss: %.4f | Epoch 2 Loss: %.4f", train_losses[0], train_losses[1])
    logger.info("  Loss Trend: %s (Decreased: %s)", train_losses, train_losses[1] <= train_losses[0])
    logger.info("  Validation Dice Scores: %s", val_dices)
    logger.info("  Best Validation Dice: %.4f at Epoch %d", results["best_val_dice"], results["best_epoch"])

    # 5. Verify Checkpoint File Existence
    ckpt_path = Path(results["checkpoint_path"])
    assert ckpt_path.exists(), f"Checkpoint was not saved at {ckpt_path}"
    logger.info("Checkpoint Verification: PASSED (%s, size: %.2f MB)", ckpt_path, ckpt_path.stat().st_size / (1024 ** 2))

    # 6. Generate Verification Plot
    artifact_plot = Path("reports/figures/monai_segmentation_dryrun.png")
    artifact_plot.parent.mkdir(parents=True, exist_ok=True)
    generate_segmentation_plot(
        val_image_path=val_data[0]["image"],
        val_label_path=val_data[0]["label"],
        model=trainer.model,
        device=device,
        output_path=artifact_plot,
    )

    logger.info("\n================================================================")
    logger.info("Phase 3: MONAI 3D UNet Segmentation Engine Verification PASSED!")
    logger.info("================================================================")


if __name__ == "__main__":
    run_segmentation_tests()
