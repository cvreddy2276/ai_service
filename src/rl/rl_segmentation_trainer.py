"""RL-Guided 3D UNet Tumor Segmentation Trainer.

Co-trains a MONAI 3D Residual UNet with a Deep Q-Network (DQN) patch-sampling
and augmentation curriculum agent on 3D DCE-MRI subtraction volumes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
)
from monai.utils import set_determinism

from src.rl.environment import ActionInterpreter, RLTrainingEnvironment, SamplingActionPolicy
from src.rl.patch_sampling_agent import AgentConfig, DQNPatchSamplingAgent
from src.segmentation.model import build_3d_unet

logger: logging.Logger = logging.getLogger(__name__)


def build_rl_segmentation_transforms(
    stage: str = "train",
    spatial_size: Tuple[int, int, int] = (96, 96, 32),
    num_samples: int = 2,
    pos: float = 1.0,
    neg: float = 1.0,
    aug_prob: float = 0.5,
    rotation_range: float = 0.3,
) -> Compose:
    """Build dynamic MONAI transforms customized by the RL agent.

    Args:
        stage: Pipeline stage ('train' or 'val').
        spatial_size: Patch size (H, W, D).
        num_samples: Number of patches per volume in training.
        pos: Foreground sampling ratio chosen by RL agent.
        neg: Background sampling ratio chosen by RL agent.
        aug_prob: Augmentation probability chosen by RL agent.
        rotation_range: Maximum rotation radians chosen by RL agent.

    Returns:
        Compose: Composed MONAI transform pipeline.
    """
    keys: Tuple[str, str] = ("image", "label")
    all_keys: List[str] = list(keys)

    transforms_list: List[Callable] = [
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=False, allow_missing_keys=True),
        Orientationd(keys=all_keys, axcodes="RAS", allow_missing_keys=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    if stage.lower() == "train":
        transforms_list.extend([
            RandCropByPosNegLabeld(
                keys=all_keys,
                label_key="label",
                spatial_size=spatial_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key="image",
            ),
            RandRotated(
                keys=all_keys,
                range_x=rotation_range,
                range_y=rotation_range,
                range_z=rotation_range,
                prob=aug_prob,
                mode=("bilinear", "nearest"),
            ),
            RandFlipd(keys=all_keys, prob=aug_prob, spatial_axis=[0, 1, 2]),
        ])

    transforms_list.extend([
        CastToTyped(keys=all_keys, dtype=torch.float32),
        EnsureTyped(keys=all_keys),
    ])

    return Compose(transforms_list)


class RLSegmentationTrainer:
    """Co-trains a MONAI 3D Residual UNet alongside a DQN Active Patch Sampling Agent."""

    def __init__(
        self,
        train_data: Sequence[Dict[str, Any]],
        val_data: Sequence[Dict[str, Any]],
        model: Optional[nn.Module] = None,
        agent: Optional[DQNPatchSamplingAgent] = None,
        agent_config: Optional[AgentConfig] = None,
        spatial_size: Tuple[int, int, int] = (96, 96, 32),
        batch_size: int = 2,
        num_samples: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        sw_batch_size: int = 4,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        checkpoint_name: str = "best_rl_monai_3dunet.pt",
        agent_checkpoint_name: str = "best_rl_segmentation_agent.pt",
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        """Initialize RLSegmentationTrainer.

        Args:
            train_data: List of training sample dictionaries ({'image': path, 'label': path}).
            val_data: List of validation sample dictionaries.
            model: Optional pre-built 3D segmentation network.
            agent: Optional pre-built DQN agent.
            agent_config: Hyperparameters for the DQN agent if agent is None.
            spatial_size: Spatial patch dimensions (H, W, D).
            batch_size: Mini-batch size.
            num_samples: Patches sampled per volume.
            lr: Learning rate for 3D UNet AdamW optimizer.
            weight_decay: Weight decay for 3D UNet optimizer.
            sw_batch_size: Sliding window batch size during full-volume validation.
            checkpoint_dir: Directory to save model checkpoints.
            checkpoint_name: Filename for best 3D UNet model.
            agent_checkpoint_name: Filename for best DQN agent.
            device: Compute device (CPU or CUDA).
            seed: Random seed for determinism.
        """
        set_determinism(seed=seed)
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_data: Sequence[Dict[str, Any]] = train_data
        self.val_data: Sequence[Dict[str, Any]] = val_data
        self.spatial_size: Tuple[int, int, int] = spatial_size
        self.batch_size: int = batch_size
        self.num_samples: int = num_samples
        self.sw_batch_size: int = sw_batch_size

        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_checkpoint_path: Path = self.checkpoint_dir / checkpoint_name
        self.agent_checkpoint_path: Path = self.checkpoint_dir / agent_checkpoint_name

        # 3D UNet Model
        self.model: nn.Module = (
            model.to(self.device) if model is not None else build_3d_unet(in_channels=1, out_channels=1, device=self.device)
        )

        # RL DQN Agent
        self.agent: DQNPatchSamplingAgent = agent or DQNPatchSamplingAgent(
            config=agent_config or AgentConfig(batch_size=4, replay_capacity=1000),
            device=self.device,
        )

        # Loss and Optimizer
        self.loss_fn: nn.Module = DiceCELoss(
            sigmoid=True,
            squared_pred=True,
            include_background=False,
        )
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Validation Metrics
        self.dice_metric: DiceMetric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric: HausdorffDistanceMetric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="mean",
        )

        # Static Validation Loader
        val_transforms = build_rl_segmentation_transforms(stage="val")
        val_dataset = Dataset(data=self.val_data, transform=val_transforms)
        self.val_loader: DataLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        self.best_val_dice: float = -1.0
        self.best_epoch: int = -1
        self.history: Dict[str, List[Any]] = {
            "train_loss": [],
            "val_dice": [],
            "val_hd95": [],
            "lr": [],
            "rl_action": [],
            "rl_action_name": [],
            "rl_reward": [],
            "rl_q_loss": [],
            "rl_epsilon": [],
        }

    def _create_train_loader(self, policy: SamplingActionPolicy) -> DataLoader:
        """Construct training DataLoader with RL-specified sampling policy.

        Args:
            policy: Action policy containing foreground/background ratios and augmentation values.

        Returns:
            DataLoader: Configured PyTorch DataLoader for the epoch.
        """
        transforms = build_rl_segmentation_transforms(
            stage="train",
            spatial_size=self.spatial_size,
            num_samples=self.num_samples,
            pos=policy.pos_weight,
            neg=policy.neg_weight,
            aug_prob=policy.aug_prob,
            rotation_range=policy.rotation_range,
        )
        dataset = Dataset(data=self.train_data, transform=transforms)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch over all patches.

        Args:
            train_loader: Configured DataLoader for the current epoch.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        epoch_loss: float = 0.0
        step_count: int = 0

        for batch_data in train_loader:
            step_count += 1
            inputs: torch.Tensor = batch_data["image"].to(self.device)
            labels: torch.Tensor = batch_data["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += float(loss.item())

        return epoch_loss / max(1, step_count)

    def validate(self) -> Tuple[float, float]:
        """Run validation using full-volume sliding-window inference.

        Returns:
            Tuple[float, float]: (Mean Dice Score, Mean HD95 in mm).
        """
        self.model.eval()
        self.dice_metric.reset()
        self.hd95_metric.reset()

        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs: torch.Tensor = val_data["image"].to(self.device)
                val_labels: torch.Tensor = val_data["label"].to(self.device)

                val_outputs: torch.Tensor = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=self.spatial_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=0.25,
                )
                val_preds: torch.Tensor = (torch.sigmoid(val_outputs) > 0.5).float()

                self.dice_metric(y_pred=val_preds, y=val_labels)
                try:
                    self.hd95_metric(y_pred=val_preds, y=val_labels)
                except Exception as e:
                    logger.debug("HD95 skipped for sample: %s", e)

        mean_dice: float = float(self.dice_metric.aggregate().item())
        try:
            mean_hd95: float = float(self.hd95_metric.aggregate().item())
        except Exception:
            mean_hd95 = float("nan")

        self.dice_metric.reset()
        self.hd95_metric.reset()

        return mean_dice, mean_hd95

    def fit(
        self,
        max_epochs: int = 10,
        val_interval: int = 1,
    ) -> Dict[str, Any]:
        """Execute the joint RL and 3D UNet training loop.

        Args:
            max_epochs: Total number of training epochs.
            val_interval: Validation frequency in epochs.

        Returns:
            Dict[str, Any]: Training summary including metrics, rewards, and paths.
        """
        logger.info(
            "Starting RL-Guided 3D UNet Segmentation Training for %d epochs on device: %s",
            max_epochs,
            self.device,
        )

        env = RLTrainingEnvironment(task="segmentation", max_epochs=max_epochs)
        state = env.reset()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )

        for epoch in range(1, max_epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 1. RL Agent selects action from state
            action: int = self.agent.select_action(state, training=True)
            action_name: str = self.agent.get_action_name(action)
            policy: SamplingActionPolicy = env.interpreter.apply_action(action)

            # 2. Build training DataLoader with RL-guided policy
            epoch_loader = self._create_train_loader(policy)

            # 3. Train 3D UNet for one epoch
            train_loss = self.train_epoch(epoch_loader)

            # 4. Validate
            if epoch % val_interval == 0:
                val_dice, val_hd95 = self.validate()
            else:
                val_dice = self.history["val_dice"][-1] if self.history["val_dice"] else 0.0
                val_hd95 = self.history["val_hd95"][-1] if self.history["val_hd95"] else float("nan")

            # 5. Environment step & Reward computation
            next_state, reward, done, info = env.step(
                action=action,
                train_loss=train_loss,
                val_metric=val_dice,
                epoch=epoch,
            )

            # 6. DQN Replay memory & Bellman learning
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
            self.history["val_dice"].append(val_dice)
            self.history["val_hd95"].append(val_hd95)
            self.history["lr"].append(current_lr)
            self.history["rl_action"].append(action)
            self.history["rl_action_name"].append(action_name)
            self.history["rl_reward"].append(reward)
            self.history["rl_q_loss"].append(q_loss if q_loss is not None else 0.0)
            self.history["rl_epsilon"].append(self.agent.epsilon)

            # Checkpoint best models
            is_best: bool = val_dice > self.best_val_dice
            log_suffix = ""
            if is_best:
                self.best_val_dice = val_dice
                self.best_epoch = epoch

                # Save UNet checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_dice": val_dice,
                        "val_hd95": val_hd95,
                        "history": self.history,
                    },
                    self.model_checkpoint_path,
                )

                # Save DQN agent checkpoint
                self.agent.save_checkpoint(
                    self.agent_checkpoint_path,
                    metadata={"best_val_dice": val_dice, "epoch": epoch},
                )
                log_suffix = " [Best Checkpoint Saved!]"

            logger.info(
                "Epoch [%02d/%02d] - Loss: %.4f | Val Dice: %.4f | RL Action: %s | Reward: %.4f | Epsilon: %.3f%s",
                epoch,
                max_epochs,
                train_loss,
                val_dice,
                action_name,
                reward,
                self.agent.epsilon,
                log_suffix,
            )

            scheduler.step()

        logger.info(
            "RL-Guided Segmentation Complete. Best Val Dice: %.4f at Epoch %d. Checkpoint: %s",
            self.best_val_dice,
            self.best_epoch,
            self.model_checkpoint_path,
        )

        return {
            "best_val_dice": self.best_val_dice,
            "best_epoch": self.best_epoch,
            "model_checkpoint_path": str(self.model_checkpoint_path),
            "agent_checkpoint_path": str(self.agent_checkpoint_path),
            "history": self.history,
        }
