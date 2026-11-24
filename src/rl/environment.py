"""Reinforcement Learning Environment and Policy Interpreter for 3D DCE-MRI Training.

Provides environment abstractions connecting discrete DQN agent actions
(patch sampling focus and augmentation curriculum) to MONAI/PyTorch volumetric pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.rl.patch_sampling_agent import DQNPatchSamplingAgent

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SamplingActionPolicy:
    """Active sampling and augmentation parameters determined by the RL agent.

    Attributes:
        action_index: Integer index of the selected action [0..4].
        action_name: Semantic name of the action.
        pos_weight: MONAI RandCrop positive (foreground) sampling weight.
        neg_weight: MONAI RandCrop negative (background) sampling weight.
        aug_prob: Spatial augmentation activation probability in [0.0, 1.0].
        rotation_range: Maximum random 3D rotation radians.
        roi_scale_factor: Bounding box scale factor for ROI crop [0.75, 1.35].
    """

    action_index: int
    action_name: str
    pos_weight: float
    neg_weight: float
    aug_prob: float
    rotation_range: float
    roi_scale_factor: float


class ActionInterpreter:
    """Translates discrete agent actions into dynamic dataset and transform policies."""

    def __init__(
        self,
        base_aug_prob: float = 0.4,
        base_rotation_range: float = 0.3,
        min_aug_prob: float = 0.1,
        max_aug_prob: float = 0.85,
        min_rotation_range: float = 0.1,
        max_rotation_range: float = 0.6,
    ) -> None:
        """Initialize the ActionInterpreter.

        Args:
            base_aug_prob: Initial augmentation probability.
            base_rotation_range: Initial 3D rotation radian limit.
            min_aug_prob: Lower bound on augmentation probability.
            max_aug_prob: Upper bound on augmentation probability.
            min_rotation_range: Lower bound on rotation angle.
            max_rotation_range: Upper bound on rotation angle.
        """
        self.aug_prob: float = base_aug_prob
        self.rotation_range: float = base_rotation_range
        self.min_aug_prob: float = min_aug_prob
        self.max_aug_prob: float = max_aug_prob
        self.min_rotation_range: float = min_rotation_range
        self.max_rotation_range: float = max_rotation_range

        # Default foreground / background weights
        self.pos_weight: float = 1.0
        self.neg_weight: float = 1.0
        self.roi_scale_factor: float = 1.0

    def apply_action(self, action_index: int) -> SamplingActionPolicy:
        """Apply the selected action to update dynamic sampling state.

        Args:
            action_index: Integer action in range [0, 4].

        Returns:
            SamplingActionPolicy: Active policy parameters.

        Raises:
            ValueError: If action_index is invalid.
        """
        if not 0 <= action_index < len(DQNPatchSamplingAgent.ACTIONS):
            raise ValueError(f"Invalid action index: {action_index}")

        action_name: str = DQNPatchSamplingAgent.ACTIONS[action_index]

        if action_name == "tumor_center_patch":
            # High foreground weighting focused directly on tumor core
            self.pos_weight = 1.0
            self.neg_weight = 0.05
            self.roi_scale_factor = 0.85
        elif action_name == "lesion_boundary_patch":
            # Balanced foreground-background boundary transition zone
            self.pos_weight = 0.60
            self.neg_weight = 0.40
            self.roi_scale_factor = 1.00
        elif action_name == "background_patch":
            # Hard negative mining of healthy parenchyma and chest wall
            self.pos_weight = 0.05
            self.neg_weight = 0.95
            self.roi_scale_factor = 1.25
        elif action_name == "increase_augmentation":
            # Escalate spatial perturbations to prevent overfitting
            self.aug_prob = min(self.max_aug_prob, self.aug_prob + 0.15)
            self.rotation_range = min(self.max_rotation_range, self.rotation_range + 0.10)
        elif action_name == "decrease_augmentation":
            # Lower perturbations to stabilize gradient descent
            self.aug_prob = max(self.min_aug_prob, self.aug_prob - 0.15)
            self.rotation_range = max(self.min_rotation_range, self.rotation_range - 0.10)

        logger.debug(
            "Action applied: [%s] -> pos=%.2f, neg=%.2f, aug_prob=%.2f, rot=%.2f, roi_scale=%.2f",
            action_name,
            self.pos_weight,
            self.neg_weight,
            self.aug_prob,
            self.rotation_range,
            self.roi_scale_factor,
        )

        return SamplingActionPolicy(
            action_index=action_index,
            action_name=action_name,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            aug_prob=self.aug_prob,
            rotation_range=self.rotation_range,
            roi_scale_factor=self.roi_scale_factor,
        )

    def get_policy_factor(self) -> float:
        """Return scalar representation of current policy intensity in [0.0, 1.0]."""
        # Composite of augmentation probability and foreground emphasis
        return float(0.5 * self.aug_prob + 0.5 * (self.pos_weight / (self.pos_weight + self.neg_weight + 1e-7)))


class RLTrainingEnvironment:
    """Environment maintaining state and calculating rewards for the RL Agent."""

    def __init__(
        self,
        task: str = "segmentation",
        max_epochs: int = 10,
    ) -> None:
        """Initialize RL training environment.

        Args:
            task: Task type ('segmentation' or 'classification').
            max_epochs: Maximum epochs in the training schedule.

        Raises:
            ValueError: If task is not 'segmentation' or 'classification'.
        """
        if task.lower() not in ("segmentation", "classification"):
            raise ValueError(f"Unsupported task: {task}. Must be 'segmentation' or 'classification'.")

        self.task: str = task.lower()
        self.max_epochs: int = max_epochs
        self.interpreter: ActionInterpreter = ActionInterpreter()

        self.prev_loss: Optional[float] = None
        self.prev_metric: Optional[float] = None
        self.current_epoch: int = 0
        self.loss_history: List[float] = []
        self.metric_history: List[float] = []

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state at episode start.

        Returns:
            torch.Tensor: Initial 5D state vector.
        """
        self.prev_loss = None
        self.prev_metric = None
        self.current_epoch = 0
        self.loss_history = []
        self.metric_history = []
        self.interpreter = ActionInterpreter()

        initial_state: torch.Tensor = DQNPatchSamplingAgent.construct_state_vector(
            train_loss=1.0,
            val_metric=0.0,
            loss_slope=0.0,
            progress=0.0,
            policy_factor=self.interpreter.get_policy_factor(),
        )
        return initial_state

    def step(
        self,
        action: int,
        train_loss: float,
        val_metric: float,
        epoch: int,
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Advance the training environment by one epoch.

        Args:
            action: Action selected by the agent.
            train_loss: Epoch average training loss.
            val_metric: Validation metric (Dice for segmentation, AUC for classification).
            epoch: Current 1-based epoch index.

        Returns:
            Tuple of:
                - next_state (torch.Tensor shaped (5,))
                - reward (float)
                - done (bool)
                - info (Dict[str, Any])
        """
        self.current_epoch = epoch
        self.loss_history.append(train_loss)
        self.metric_history.append(val_metric)

        # Compute metric and loss deltas
        if self.prev_metric is None:
            metric_delta = 0.0
        else:
            metric_delta = val_metric - self.prev_metric

        if self.prev_loss is None:
            loss_delta = 0.0
        else:
            loss_delta = train_loss - self.prev_loss

        # Calculate reward
        if self.task == "segmentation":
            reward = DQNPatchSamplingAgent.compute_reward(
                dice_delta=metric_delta,
                auroc_delta=0.0,
                loss_delta=loss_delta,
            )
        else:
            reward = DQNPatchSamplingAgent.compute_reward(
                dice_delta=0.0,
                auroc_delta=metric_delta,
                loss_delta=loss_delta,
            )

        self.prev_loss = train_loss
        self.prev_metric = val_metric

        # Update policy from action
        policy: SamplingActionPolicy = self.interpreter.apply_action(action)

        # Calculate progress and loss slope
        progress = float(min(1.0, epoch / max(1, self.max_epochs)))
        loss_slope = float(loss_delta)
        policy_factor = self.interpreter.get_policy_factor()

        next_state: torch.Tensor = DQNPatchSamplingAgent.construct_state_vector(
            train_loss=train_loss,
            val_metric=val_metric,
            loss_slope=loss_slope,
            progress=progress,
            policy_factor=policy_factor,
        )

        done: bool = epoch >= self.max_epochs
        info: Dict[str, Any] = {
            "epoch": epoch,
            "action_name": policy.action_name,
            "metric_delta": metric_delta,
            "loss_delta": loss_delta,
            "reward": reward,
            "policy": policy,
        }

        return next_state, reward, done, info
