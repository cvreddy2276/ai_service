"""Phase 5 Verification Script: 3D Deep Learning Classification & Evaluation Pipeline.

Validates end-to-end 3D Deep Learning response prediction (pCR) on volumetric DCE-MRI tensors:
- Instantiates 3D Convolutional Neural Network (SimpleCNN3D / ResNet3D / DenseNet3D)
- Executes multi-epoch PyTorch training pass with learning rate scheduling
- Evaluates validation ROC-AUC, Balanced Accuracy, Sensitivity, Specificity, and Confusion Matrix
- Exports diagnostic training curves and ROC figures
- Verifies checkpoint saving to checkpoints/best_3d_classifier.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.classification.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_single_roc_curve,
    plot_training_curves,
    save_classification_report,
)
from src.classification.trainer import DCE3DClassificationTrainer
from src.models.classifier3d import build_3d_classifier
from src.utils.dummy_generator import create_synthetic_dce_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Test3DClassification")

REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_synthetic_3d_dataset(
    num_samples: int = 16,
    spatial_size: tuple[int, int, int] = (64, 64, 24),
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Generate synthetic 3D volumetric DCE-MRI tensors for classification testing.

    Args:
        num_samples: Total number of 3D volumetric samples to synthesize.
        spatial_size: (H, W, D) spatial dimension of each 3D volume.
        seed: Random seed for reproducibility.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Balanced binary labels (0 = non-pCR, 1 = pCR)
    labels = np.array([i % 2 for i in range(num_samples)], dtype=np.float32)

    volumes: List[torch.Tensor] = []
    for i in range(num_samples):
        # Base background noise
        vol = torch.randn(1, *spatial_size) * 0.1
        # If class 1 (responder), inject enhanced focal hotspot in center
        if labels[i] == 1.0:
            cz, cy, cx = spatial_size[2] // 2, spatial_size[1] // 2, spatial_size[0] // 2
            vol[:, cx - 6:cx + 6, cy - 6:cy + 6, cz - 3:cz + 3] += 1.5
        volumes.append(vol)

    vol_tensor = torch.stack(volumes)
    label_tensor = torch.from_numpy(labels).unsqueeze(1)

    # Train / Val Split (75% train, 25% val)
    split_idx = int(num_samples * 0.75)
    train_ds = TensorDataset(vol_tensor[:split_idx], label_tensor[:split_idx])
    val_ds = TensorDataset(vol_tensor[split_idx:], label_tensor[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    return train_loader, val_loader


def run_classification_tests() -> None:
    """Execute complete Phase 5: 3D Deep Learning Classification Verification."""
    logger.info("================================================================")
    logger.info("Starting Phase 5: 3D Deep Learning Classification Verification")
    logger.info("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Operating PyTorch Device: %s", device)

    # 1. Instantiate 3D Deep Learning Model
    logger.info("\n--- 1. Instantiating 3D Deep Convolutional Classifier ---")
    model = build_3d_classifier(
        spatial_dims=3,
        in_channels=1,
        num_classes=1,
        backbone="simple_cnn3d",
        dropout=0.2,
        device=device,
    )

    # Verify forward pass on dummy 3D batch
    dummy_input = torch.randn(2, 1, 64, 64, 24, device=device)
    with torch.no_grad():
        dummy_out = model(dummy_input)
    logger.info(
        "Forward pass verified: Input %s -> Output Logits %s",
        tuple(dummy_input.shape),
        tuple(dummy_out.shape),
    )
    assert dummy_out.shape == (2, 1), f"Expected shape (2, 1), got {dummy_out.shape}"

    # 2. Prepare 3D Volumetric DataLoaders
    logger.info("\n--- 2. Preparing 3D DCE-MRI Volumetric DataLoaders ---")
    train_loader, val_loader = prepare_synthetic_3d_dataset(
        num_samples=16,
        spatial_size=(64, 64, 24),
        seed=42,
    )
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # 3. Train 3D Deep Learning Classifier
    logger.info("\n--- 3. Executing 3D Deep Learning Training Loop ---")
    checkpoint_path = CHECKPOINTS_DIR / "best_3d_classifier.pt"
    trainer = DCE3DClassificationTrainer(
        model=model,
        lr=5e-4,
        weight_decay=1e-3,
        device=device,
    )

    epochs = 6
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        image_key="0",
        label_key="1",
    )

    # Verify Loss Decreased
    assert len(history["train_loss"]) == epochs, "Training history incomplete"
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    logger.info("Initial Train Loss: %.4f -> Final Train Loss: %.4f", initial_loss, final_loss)
    assert final_loss < initial_loss, "Training loss did not decrease over epochs"

    # 4. Final Validation & Metric Computation
    logger.info("\n--- 4. Final Model Validation & Checkpoint Loading ---")
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"
    trainer.load_checkpoint(checkpoint_path)

    val_loss, val_metrics, y_true, y_prob = trainer.validate(
        val_loader,
        image_key="0",
        label_key="1",
    )

    logger.info("\n--- 3D DEEP LEARNING VALIDATION RESULTS ---")
    for metric_name, val in val_metrics.items():
        if isinstance(val, float):
            logger.info("  %-20s: %.4f", metric_name, val)
        else:
            logger.info("  %-20s: %d", metric_name, val)

    # 5. Export Diagnostic Figures & Report
    logger.info("\n--- 5. Generating Diagnostic Figures & Classification Report ---")

    # Training Loss & AUC curves
    train_curves_path = REPORTS_DIR / "dl_training_curves.png"
    plot_training_curves(history, train_curves_path)

    # ROC curve
    roc_curve_path = REPORTS_DIR / "dl_roc_curve.png"
    plot_single_roc_curve(y_true, y_prob, roc_curve_path, model_label="3D Deep ConvNet")

    # Confusion matrix
    y_pred = (y_prob >= 0.5).astype(int)
    cm_path = REPORTS_DIR / "dl_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, title="3D Deep Learning Confusion Matrix")

    # Save JSON Report
    report_data = {
        "3d_deep_convnet": {
            **val_metrics,
            "val_loss": val_loss,
            "epochs_trained": epochs,
            "device": str(device),
            "checkpoint": str(checkpoint_path),
        }
    }
    report_json_path = Path("reports/classification_report.json")
    save_classification_report(report_data, report_json_path)

    logger.info("\n--- SAVED OUTPUTS ---")
    for fig_f in sorted(REPORTS_DIR.glob("dl_*.png")):
        logger.info("  %s (%.1f KB)", fig_f, fig_f.stat().st_size / 1024.0)
    logger.info("  %s (%.1f KB)", checkpoint_path, checkpoint_path.stat().st_size / 1024.0)
    logger.info("  %s (%.1f KB)", report_json_path, report_json_path.stat().st_size / 1024.0)

    logger.info("\n================================================================")
    logger.info("Phase 5: 3D Deep Learning Classification Verification PASSED!")
    logger.info("================================================================")


if __name__ == "__main__":
    run_classification_tests()
