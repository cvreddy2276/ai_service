"""RL-Guided 3D DCE-MRI Breast Cancer Classification Trainer.

Co-trains a 3D volumetric convolutional neural network (DenseNet3D / ResNet3D / SimpleCNN3D)
with a Deep Q-Network (DQN) active ROI context and augmentation curriculum agent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset
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
from monai.utils import set_determinism

from src.classification.evaluate import compute_classification_metrics
from src.models.classifier3d import build_3d_classifier
from src.preprocessing.transforms import (
    ComputeSubtractiond,
    N4BiasFieldCorrectiond,
    TumorRoiCropd,
    ZScoreNormalizeIntensityd,
)
from src.rl.environment import ActionInterpreter, RLTrainingEnvironment, SamplingActionPolicy
from src.rl.patch_sampling_agent import AgentConfig, DQNPatchSamplingAgent

logger: logging.Logger = logging.getLogger(__name__)


def build_rl_classification_transforms(
    roi_size: Tuple[int, int, int] = (128, 128, 64),
    is_training: bool = False,
    aug_prob: float = 0.4,
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    keys: Sequence[str] = ("t0", "t1", "mask"),
) -> Compose:
    """Construct dynamic classification transforms influenced by RL policy.

    Args:
        roi_size: Dynamic spatial crop dimensions (H, W, D) adjusted by RL agent.
        is_training: Whether training augmentations should be active.
        aug_prob: Probability of spatial and intensity augmentations.
        pixdim: Target isotropic voxel spacing.
        keys: Dictionary keys for multi-temporal DCE volumes and mask.

    Returns:
        Compose: Configured MONAI transform pipeline.
    """
    image_keys: List[str] = [k for k in keys if k != "mask"]
    all_keys: List[str] = list(keys)

    transforms_list: List[Callable] = [
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=False, allow_missing_keys=True),
        Orientationd(keys=all_keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=all_keys,
            pixdim=pixdim,
            mode=["bilinear" if k in image_keys else "nearest" for k in all_keys],
            allow_missing_keys=True,
        ),
        ComputeSubtractiond(
            pre_key="t0",
            post_key="t1",
            sub_key="subtraction",
            multichannel_key="image",
            clip_negative=True,
            allow_missing_keys=True,
        ),
        ZScoreNormalizeIntensityd(
            keys=image_keys + ["subtraction", "image"],
            channel_wise=True,
            nonzero=True,
            allow_missing_keys=True,
        ),
        TumorRoiCropd(
            keys=["image", "t0", "t1", "subtraction", "mask"],
            roi_size=roi_size,
            mask_key="mask" if "mask" in all_keys else None,
            allow_missing_keys=True,
        ),
    ]

    if is_training:
        transforms_list.extend([
            RandFlipd(keys=["image"], prob=aug_prob, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=["image"], prob=aug_prob, spatial_axis=1, allow_missing_keys=True),
            RandRotate90d(keys=["image"], prob=aug_prob, max_k=3, allow_missing_keys=True),
            RandGaussianNoised(keys=["image"], prob=min(aug_prob * 0.5, 0.3), mean=0.0, std=0.08, allow_missing_keys=True),
        ])

    transforms_list.extend([
        CastToTyped(keys=["image", "subtraction"], dtype=torch.float32, allow_missing_keys=True),
        EnsureTyped(keys=["image"], allow_missing_keys=True),
    ])

    return Compose(transforms_list)


class RLClassificationTrainer:
    """Co-trains a 3D DCE-MRI Classifier with a DQN Active Sampling / Curriculum Agent."""

    def __init__(
        self,
        train_samples: Optional[Sequence[Dict[str, Any]]] = None,
        val_samples: Optional[Sequence[Dict[str, Any]]] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        model: Optional[nn.Module] = None,
        backbone: str = "simple_cnn3d",
        in_channels: int = 1,
        num_classes: int = 1,
        base_roi_size: Tuple[int, int, int] = (128, 128, 64),
        agent: Optional[DQNPatchSamplingAgent] = None,
        agent_config: Optional[AgentConfig] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        batch_size: int = 2,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        checkpoint_name: str = "best_rl_3d_classifier.pt",
        agent_checkpoint_name: str = "best_rl_classification_agent.pt",
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        """Initialize RLClassificationTrainer.

        Args:
            train_samples: Optional training sample dictionaries.
            val_samples: Optional validation sample dictionaries.
            train_loader: Optional pre-constructed training DataLoader.
            val_loader: Optional pre-constructed validation DataLoader.
            model: Optional pre-constructed PyTorch 3D network.
            backbone: Backbone name if model is None ('simple_cnn3d', 'resnet18', 'densenet121').
            in_channels: Number of input channels (1 for subtraction, 3 for multi-channel).
            num_classes: Classification output dimension (1 for binary pCR).
            base_roi_size: Canonical 3D ROI crop size (H, W, D).
            agent: Optional pre-constructed DQNPatchSamplingAgent.
            agent_config: Hyperparameters for the DQN agent.
            lr: Learning rate for 3D classifier optimizer.
            weight_decay: Weight decay regularizer.
            batch_size: Mini-batch size.
            checkpoint_dir: Folder to save checkpoints.
            checkpoint_name: File name for best 3D classifier.
            agent_checkpoint_name: File name for best DQN agent.
            device: Compute device (CPU or CUDA).
            seed: Random seed.

        Raises:
            ValueError: If neither samples nor DataLoaders are provided.
        """
        set_determinism(seed=seed)
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if train_samples is None and train_loader is None:
            raise ValueError("Either train_samples or train_loader must be provided.")

        self.train_samples: Optional[Sequence[Dict[str, Any]]] = train_samples
        self.val_samples: Optional[Sequence[Dict[str, Any]]] = val_samples
        self.static_train_loader: Optional[DataLoader] = train_loader
        self.base_roi_size: Tuple[int, int, int] = base_roi_size
        self.batch_size: int = batch_size
        self.in_channels: int = in_channels

        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_checkpoint_path: Path = self.checkpoint_dir / checkpoint_name
        self.agent_checkpoint_path: Path = self.checkpoint_dir / agent_checkpoint_name

        # 3D Classifier
        if model is not None:
            self.model: nn.Module = model.to(self.device)
        else:
            self.model = build_3d_classifier(
                spatial_dims=3,
                in_channels=in_channels,
                num_classes=num_classes,
                backbone=backbone,
                device=self.device,
            )

        # DQN Agent
        self.agent: DQNPatchSamplingAgent = agent or DQNPatchSamplingAgent(
            config=agent_config or AgentConfig(batch_size=4, replay_capacity=1000),
            device=self.device,
        )

        # Loss & Optimizer
        self.criterion: nn.Module = nn.BCEWithLogitsLoss()
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Validation Loader
        if val_loader is not None:
            self.val_loader: DataLoader = val_loader
        elif self.val_samples is not None:
            val_transforms = build_rl_classification_transforms(
                roi_size=self.base_roi_size,
                is_training=False,
            )
            val_dataset = Dataset(data=self.val_samples, transform=val_transforms)
            self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        else:
            raise ValueError("Either val_samples or val_loader must be provided.")

        self.best_val_auc: float = -1.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1

        self.history: Dict[str, List[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
            "rl_action": [],
            "rl_action_name": [],
            "rl_reward": [],
            "rl_q_loss": [],
            "rl_epsilon": [],
        }

    def _compute_scaled_roi_size(self, scale_factor: float) -> Tuple[int, int, int]:
        """Scale base ROI dimensions ensuring multiples of 16 for deep architectures."""
        scaled = [
            int(round(dim * scale_factor / 16.0) * 16)
            for dim in self.base_roi_size
        ]
        return (max(32, scaled[0]), max(32, scaled[1]), max(16, scaled[2]))

    def _create_train_loader(self, policy: SamplingActionPolicy) -> DataLoader:
        """Create training DataLoader dynamically scaled by RL policy."""
        if self.train_samples is None:
            raise RuntimeError("train_samples is None; cannot create dynamic train loader.")

        scaled_roi = self._compute_scaled_roi_size(policy.roi_scale_factor)
        transforms = build_rl_classification_transforms(
            roi_size=scaled_roi,
            is_training=True,
            aug_prob=policy.aug_prob,
        )
        dataset = Dataset(data=self.train_samples, transform=transforms)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train classifier for one epoch over all batches.

        Args:
            train_loader: Dynamic training DataLoader.

        Returns:
            float: Mean training loss.
        """
        self.model.train()
        total_loss: float = 0.0
        num_batches: int = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                if self.in_channels == 1 and "subtraction" in batch:
                    img_val = batch["subtraction"]
                else:
                    img_val = batch.get("image", batch.get("subtraction"))
                inputs: torch.Tensor = img_val.to(self.device, dtype=torch.float32)
                targets: torch.Tensor = batch["label"].to(self.device, dtype=torch.float32)
            else:
                inputs, targets = batch[0].to(self.device, dtype=torch.float32), batch[1].to(self.device, dtype=torch.float32)

            if self.in_channels == 1 and inputs.shape[1] > 1:
                inputs = inputs[:, 2:3] if inputs.shape[1] >= 3 else inputs[:, 0:1]
            elif self.in_channels == 3 and inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1, 1)

            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            logits: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate classifier on validation set and calculate clinical metrics.

        Returns:
            Tuple of (average_val_loss, metrics_dictionary).
        """
        self.model.eval()
        total_loss: float = 0.0
        num_batches: int = 0

        y_true_list: List[float] = []
        y_prob_list: List[float] = []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    if self.in_channels == 1 and "subtraction" in batch:
                        img_val = batch["subtraction"]
                    else:
                        img_val = batch.get("image", batch.get("subtraction"))
                    inputs: torch.Tensor = img_val.to(self.device, dtype=torch.float32)
                    targets: torch.Tensor = batch["label"].to(self.device, dtype=torch.float32)
                else:
                    inputs, targets = batch[0].to(self.device, dtype=torch.float32), batch[1].to(self.device, dtype=torch.float32)

                if self.in_channels == 1 and inputs.shape[1] > 1:
                    inputs = inputs[:, 2:3] if inputs.shape[1] >= 3 else inputs[:, 0:1]
                elif self.in_channels == 3 and inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1, 1)

                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)

                logits: torch.Tensor = self.model(inputs)
                loss: torch.Tensor = self.criterion(logits, targets)
                probs: torch.Tensor = torch.sigmoid(logits)

                total_loss += float(loss.item())
                num_batches += 1

                y_true_list.extend(targets.squeeze(1).cpu().numpy().tolist())
                y_prob_list.extend(probs.squeeze(1).cpu().numpy().tolist())

        avg_loss: float = total_loss / max(num_batches, 1)
        y_true = np.array(y_true_list, dtype=np.int32)
        y_prob = np.array(y_prob_list, dtype=np.float32)

        metrics: Dict[str, float] = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        return avg_loss, metrics

    def fit(
        self,
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Execute co-training of 3D Classifier and DQN Agent.

        Args:
            epochs: Total epochs to run.

        Returns:
            Dict[str, Any]: History and checkpoint summaries.
        """
        logger.info(
            "Starting RL-Guided 3D Classification Training for %d epochs on device: %s",
            epochs,
            self.device,
        )

        env = RLTrainingEnvironment(task="classification", max_epochs=epochs)
        state = env.reset()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 1. RL Agent selects action
            action: int = self.agent.select_action(state, training=True)
            action_name: str = self.agent.get_action_name(action)
            policy: SamplingActionPolicy = env.interpreter.apply_action(action)

            # 2. Build training DataLoader with RL-specified crop & augmentation policy
            if self.static_train_loader is not None:
                train_loader = self.static_train_loader
            else:
                train_loader = self._create_train_loader(policy)

            # 3. Train epoch
            train_loss = self.train_epoch(train_loader)

            # 4. Validate
            val_loss, metrics = self.validate()
            val_auc = float(metrics.get("ROC_AUC", 0.5))
            val_acc = float(metrics.get("Balanced_Accuracy", 0.5))
            val_f1 = float(metrics.get("F1_Score", 0.0))

            # 5. Environment step & Reward
            next_state, reward, done, info = env.step(
                action=action,
                train_loss=train_loss,
                val_metric=val_auc,
                epoch=epoch,
            )

            # 6. DQN Bellman update
            self.agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            q_loss = self.agent.learn()
            state = next_state

            # 7. Record History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)
            self.history["rl_action"].append(action)
            self.history["rl_action_name"].append(action_name)
            self.history["rl_reward"].append(reward)
            self.history["rl_q_loss"].append(q_loss if q_loss is not None else 0.0)
            self.history["rl_epsilon"].append(self.agent.epsilon)

            is_best: bool = val_auc > self.best_val_auc or (val_auc == self.best_val_auc and val_loss < self.best_val_loss)
            log_suffix = ""

            if is_best:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.best_epoch = epoch

                # Save 3D classifier checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": metrics,
                        "history": self.history,
                    },
                    self.model_checkpoint_path,
                )

                # Save DQN agent checkpoint
                self.agent.save_checkpoint(
                    self.agent_checkpoint_path,
                    metadata={"best_val_auc": val_auc, "epoch": epoch},
                )
                log_suffix = " [Best Checkpoint Saved!]"

            logger.info(
                "Epoch [%02d/%02d] - Loss: %.4f | Val AUC: %.4f | Acc: %.4f | RL Action: %s | Reward: %.4f%s",
                epoch,
                epochs,
                train_loss,
                val_auc,
                val_acc,
                action_name,
                reward,
                log_suffix,
            )

            scheduler.step()

        logger.info(
            "RL-Guided Classification Complete. Best Val AUC: %.4f at Epoch %d. Checkpoint: %s",
            self.best_val_auc,
            self.best_epoch,
            self.model_checkpoint_path,
        )

        return {
            "best_val_auc": self.best_val_auc,
            "best_epoch": self.best_epoch,
            "model_checkpoint_path": str(self.model_checkpoint_path),
            "agent_checkpoint_path": str(self.agent_checkpoint_path),
            "history": self.history,
        }
