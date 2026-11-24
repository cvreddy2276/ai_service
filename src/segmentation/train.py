"""Training and evaluation pipeline for MONAI 3D UNet Tumor Segmentation.

Features:
- AdamW optimizer with Cosine Annealing learning rate scheduling
- Combined Dice and Cross-Entropy Loss (DiceCELoss) with sigmoid activation
- Full-volume sliding window inference during validation
- Mean Dice and 95th Percentile Hausdorff Distance (HD95) metric tracking
- Automated best model checkpointing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils import set_determinism

from src.segmentation.model import build_3d_unet

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Trainer class for 3D UNet Breast Tumor Segmentation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        roi_size: Tuple[int, int, int] = (96, 96, 32),
        sw_batch_size: int = 4,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        checkpoint_name: str = "best_monai_3dunet.pt",
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        """Initialize SegmentationTrainer.

        Args:
            model: PyTorch/MONAI 3D segmentation network.
            train_loader: Training DataLoader yielding patch dictionaries.
            val_loader: Validation DataLoader yielding full volume dictionaries.
            optimizer: Optimizer (default: AdamW with lr=1e-3, weight_decay=1e-4).
            lr_scheduler: Learning rate scheduler (default: CosineAnnealingLR).
            loss_fn: Loss function (default: DiceCELoss with sigmoid=True).
            roi_size: Sliding window patch size for validation (default: (96, 96, 32)).
            sw_batch_size: Batch size for sliding window inference (default: 4).
            checkpoint_dir: Directory to store model checkpoints.
            checkpoint_name: Filename for the best model checkpoint.
            device: Target device (CPU / CUDA).
            seed: Random seed for reproducibility.
        """
        set_determinism(seed=seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Loss function
        self.loss_fn = loss_fn or DiceCELoss(
            sigmoid=True,
            squared_pred=True,
            include_background=False,
        )

        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        # Scheduler
        self.lr_scheduler = lr_scheduler

        # Validation Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="mean",
        )

        self.best_val_dice = -1.0
        self.best_epoch = -1
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_dice": [],
            "val_hd95": [],
            "lr": [],
        }

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch over all batches.

        Args:
            epoch: Current epoch index (1-based).

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        step_count = 0

        for batch_data in self.train_loader:
            step_count += 1
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, step_count)
        return avg_loss

    def validate(self) -> Tuple[float, float]:
        """Run validation using full-volume sliding window inference.

        Returns:
            Tuple[float, float]: (Mean Dice Score, Mean 95th Percentile Hausdorff Distance).
        """
        self.model.eval()
        self.dice_metric.reset()
        self.hd95_metric.reset()

        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)

                # Sliding window inference for full 3D volumes
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=0.25,
                )

                # Post-processing: Apply sigmoid and threshold at 0.5
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()

                # Update metrics
                self.dice_metric(y_pred=val_preds, y=val_labels)
                try:
                    self.hd95_metric(y_pred=val_preds, y=val_labels)
                except Exception as e:
                    logger.debug("HD95 calculation skipped for batch: %s", e)

        mean_dice = float(self.dice_metric.aggregate().item())
        try:
            mean_hd95 = float(self.hd95_metric.aggregate().item())
        except Exception:
            mean_hd95 = float("nan")

        self.dice_metric.reset()
        self.hd95_metric.reset()

        return mean_dice, mean_hd95

    def fit(self, max_epochs: int = 10, val_interval: int = 1) -> Dict[str, Any]:
        """Execute complete training and validation loop.

        Args:
            max_epochs: Total number of training epochs.
            val_interval: Frequency of validation runs in epochs.

        Returns:
            Dict[str, Any]: Training summary including best metrics and history.
        """
        logger.info(
            "Starting 3D UNet training for %d epochs on device: %s",
            max_epochs,
            self.device,
        )

        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=1e-6,
            )

        for epoch in range(1, max_epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(current_lr)

            log_msg = f"Epoch [{epoch}/{max_epochs}] - LR: {current_lr:.2e} - Train Loss: {train_loss:.4f}"

            if epoch % val_interval == 0:
                val_dice, val_hd95 = self.validate()
                self.history["val_dice"].append(val_dice)
                self.history["val_hd95"].append(val_hd95)

                log_msg += f" | Val Dice: {val_dice:.4f} | Val HD95: {val_hd95:.2f} mm"

                # Checkpoint best model
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_val_dice": val_dice,
                            "val_hd95": val_hd95,
                            "history": self.history,
                        },
                        self.checkpoint_path,
                    )
                    log_msg += " [Best Checkpoint Saved!]"

            logger.info(log_msg)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logger.info(
            "Training completed. Best Validation Dice: %.4f at epoch %d. Model saved to %s",
            self.best_val_dice,
            self.best_epoch,
            self.checkpoint_path,
        )

        return {
            "best_val_dice": self.best_val_dice,
            "best_epoch": self.best_epoch,
            "checkpoint_path": str(self.checkpoint_path),
            "history": self.history,
        }


def train_segmentation(
    train_data: Sequence[Dict[str, Any]],
    val_data: Sequence[Dict[str, Any]],
    max_epochs: int = 10,
    batch_size: int = 2,
    spatial_size: Tuple[int, int, int] = (96, 96, 32),
    num_samples: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """High-level function to build data loaders, model, and execute training.

    Args:
        train_data: Training samples list [{'image': path, 'label': path}, ...].
        val_data: Validation samples list.
        max_epochs: Number of epochs to train.
        batch_size: Mini-batch size.
        spatial_size: Patch size for positive/negative sampling.
        num_samples: Patches sampled per volume.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        checkpoint_dir: Directory to store model checkpoint.
        device: PyTorch device.

    Returns:
        Dict[str, Any]: Training summary dictionary.
    """
    from src.segmentation.dataset import get_segmentation_dataloader

    train_loader = get_segmentation_dataloader(
        data=train_data,
        stage="train",
        batch_size=batch_size,
        spatial_size=spatial_size,
        num_samples=num_samples,
    )

    val_loader = get_segmentation_dataloader(
        data=val_data,
        stage="val",
        batch_size=1,
    )

    model = build_3d_unet(
        in_channels=1,
        out_channels=1,
        device=device,
    )

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        roi_size=spatial_size,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    results = trainer.fit(max_epochs=max_epochs)
    return results
