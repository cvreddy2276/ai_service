"""3D Deep Learning Classification Pipeline for Breast Cancer Treatment Response Prediction.

Trains 3D volumetric deep neural networks directly on 3D DCE-MRI sequences to predict
pathological Complete Response (pCR) from neoadjuvant chemotherapy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classification.evaluate import compute_classification_metrics
from src.models.classifier3d import build_3d_classifier

logger = logging.getLogger(__name__)


class DCE3DClassificationTrainer:
    """3D Deep Learning Trainer for Volumetric DCE-MRI Breast Cancer Classification.

    Trains deep convolutional neural networks (DenseNet3D, ResNet3D, SimpleCNN3D)
    using PyTorch and MONAI on 3D DCE-MRI tensors (B, C, H, W, D).

    Attributes:
        model: PyTorch 3D neural network module.
        device: PyTorch compute device (CPU or CUDA).
        criterion: Loss function (default: BCEWithLogitsLoss).
        optimizer: PyTorch optimizer (default: AdamW).
        scheduler: Optional learning rate scheduler.
        history: Dictionary tracking training and validation metrics per epoch.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        backbone: str = "simple_cnn3d",
        in_channels: int = 1,
        num_classes: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        pos_weight: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize DCE3DClassificationTrainer.

        Args:
            model: Optional pre-constructed PyTorch 3D nn.Module. If None, builds from backbone.
            backbone: Model backbone if model is None ('simple_cnn3d', 'resnet18', 'densenet121').
            in_channels: Input channels (1 for subtraction, 3 for multi-channel DCE).
            num_classes: Number of output targets (default: 1 for binary classification).
            lr: Learning rate for AdamW optimizer (default: 1e-4).
            weight_decay: L2 regularization penalty (default: 1e-2).
            pos_weight: Optional positive class weighting scalar for class imbalance.
            device: Target device (CPU or CUDA). Auto-detected if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = build_3d_classifier(
                spatial_dims=3,
                in_channels=in_channels,
                num_classes=num_classes,
                backbone=backbone,
                device=self.device,
            )

        if pos_weight is not None:
            pos_tensor = torch.tensor([pos_weight], device=self.device, dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
            "val_f1": [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        image_key: str = "image",
        label_key: str = "label",
    ) -> float:
        """Execute one training epoch.

        Args:
            train_loader: PyTorch DataLoader supplying training batches.
            image_key: Batch dictionary key for input 3D tensor (default: 'image').
            label_key: Batch dictionary key for target label (default: 'label').

        Returns:
            float: Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                inputs = batch[image_key].to(self.device, dtype=torch.float32)
                targets = batch[label_key].to(self.device, dtype=torch.float32)
            else:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader,
        image_key: str = "image",
        label_key: str = "label",
    ) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Execute validation pass and compute performance metrics.

        Args:
            val_loader: PyTorch DataLoader supplying validation batches.
            image_key: Batch dictionary key for input 3D tensor.
            label_key: Batch dictionary key for target label.

        Returns:
            Tuple of (avg_val_loss, metrics_dict, y_true_array, y_prob_array).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        y_true_list: List[float] = []
        y_prob_list: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    inputs = batch[image_key].to(self.device, dtype=torch.float32)
                    targets = batch[label_key].to(self.device, dtype=torch.float32)
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                probs = torch.sigmoid(logits)

                total_loss += loss.item()
                num_batches += 1

                y_true_list.extend(targets.squeeze(1).cpu().numpy().tolist())
                y_prob_list.extend(probs.squeeze(1).cpu().numpy().tolist())

        avg_loss = total_loss / max(num_batches, 1)
        y_true = np.array(y_true_list, dtype=np.int32)
        y_prob = np.array(y_prob_list, dtype=np.float32)

        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        return avg_loss, metrics, y_true, y_prob

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_path: Optional[Union[str, Path]] = None,
        image_key: str = "image",
        label_key: str = "label",
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Fit the 3D Deep Learning model across multiple epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Total training epochs (default: 10).
            checkpoint_path: Destination path to save best model weights.
            image_key: Key name for input volume in dictionary batch.
            label_key: Key name for target label in dictionary batch.
            early_stopping_patience: Stop after this many epochs without validation ROC-AUC
                improvement. None disables early stopping.

        Returns:
            Dict[str, List[float]]: Metric training history dictionary.
        """
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1 or None.")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        best_val_auc = -1.0
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        logger.info("Starting 3D Deep Learning training for %d epochs on device %s...", epochs, self.device)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, image_key=image_key, label_key=label_key)
            val_loss, metrics, _, _ = self.validate(val_loader, image_key=image_key, label_key=label_key)

            if self.scheduler is not None:
                self.scheduler.step()

            val_auc = metrics.get("ROC_AUC", 0.5)
            val_acc = metrics.get("Balanced_Accuracy", 0.5)
            val_f1 = metrics.get("F1_Score", 0.0)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            lr_current = self.optimizer.param_groups[0]["lr"]
            is_best = val_auc > best_val_auc or (val_auc == best_val_auc and val_loss < best_val_loss)

            log_msg = (
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"LR: {lr_current:.2e} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            if is_best:
                best_val_auc = val_auc
                best_val_loss = val_loss
                epochs_without_improvement = 0
                log_msg += " [Best Checkpoint Saved!]"
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, epoch=epoch, metrics=metrics)
            else:
                epochs_without_improvement += 1

            logger.info(log_msg)
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                logger.info(
                    "Early stopping after %d epochs without validation ROC-AUC improvement.",
                    epochs_without_improvement,
                )
                break

        logger.info(
            "Training completed. Best Validation AUC: %.4f. Saved to %s",
            best_val_auc,
            checkpoint_path,
        )
        return self.history

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save PyTorch model weights and training metadata.

        Args:
            checkpoint_path: Destination file path (.pt).
            epoch: Epoch number at checkpoint.
            metrics: Validation metrics dictionary.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(payload, str(checkpoint_path))
        logger.debug("Checkpoint saved to %s", checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load PyTorch model weights from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        payload = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            self.model.load_state_dict(payload["model_state_dict"])
        else:
            self.model.load_state_dict(payload)
        logger.info("Loaded 3D classifier checkpoint from %s", checkpoint_path)
