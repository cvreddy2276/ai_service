"""Deep reinforcement-learning controller for 3D DCE-MRI training."""

from __future__ import annotations

import logging
from pathlib import Path
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the DQN patch-sampling agent."""

    state_dim: int = 5
    action_dim: int = 5
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_capacity: int = 10_000
    batch_size: int = 32
    target_update_frequency: int = 100


@dataclass(frozen=True)
class Transition:
    """One agent transition."""

    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class QNetwork(nn.Module):
    """Multilayer perceptron estimating action values."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        """Initialize the Q-network.

        Args:
            state_dim: Number of normalized state features.
            action_dim: Number of available actions.
            hidden_dim: Hidden-layer width.

        Raises:
            ValueError: If any dimension is not positive.
        """
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Network dimensions must be positive.")

        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values.

        Args:
            state: Tensor shaped `(B, state_dim)`.

        Returns:
            Tensor shaped `(B, action_dim)`.
        """
        return self.network(state)


class DQNPatchSamplingAgent:
    """DQN agent for selecting medically meaningful training actions."""

    ACTIONS: Tuple[str, ...] = (
        "tumor_center_patch",
        "lesion_boundary_patch",
        "background_patch",
        "increase_augmentation",
        "decrease_augmentation",
    )

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize online and target Q-networks.

        Args:
            config: Agent hyperparameters.
            device: Torch device.

        Raises:
            ValueError: If the configured action count is invalid.
        """
        self.config: AgentConfig = config or AgentConfig()
        if self.config.action_dim != len(self.ACTIONS):
            raise ValueError("action_dim must equal the number of defined actions.")

        self.device: torch.device = device or torch.device("cpu")
        self.online_network: QNetwork = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
        ).to(self.device)
        self.target_network: QNetwork = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay_buffer: Deque[Transition] = deque(
            maxlen=self.config.replay_capacity
        )
        self.epsilon: float = self.config.epsilon_start
        self.steps: int = 0

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select an action using epsilon-greedy exploration.

        Args:
            state: Tensor shaped `(state_dim,)` or `(1, state_dim)`.
            training: Whether exploration is enabled.

        Returns:
            Integer action index.

        Raises:
            ValueError: If the state has an invalid shape.
        """
        if state.numel() != self.config.state_dim:
            raise ValueError(
                f"Expected {self.config.state_dim} state values, got {state.numel()}."
            )

        if training and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)

        with torch.no_grad():
            state_batch: torch.Tensor = state.reshape(1, -1).to(self.device)
            q_values: torch.Tensor = self.online_network(state_batch)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store a transition.

        Args:
            state: Current state shaped `(state_dim,)`.
            action: Selected action index.
            reward: Scalar reward.
            next_state: Next state shaped `(state_dim,)`.
            done: Episode termination flag.

        Raises:
            ValueError: If the action or state dimensions are invalid.
        """
        if not 0 <= action < self.config.action_dim:
            raise ValueError(f"Invalid action index: {action}.")
        if state.numel() != self.config.state_dim:
            raise ValueError("State has an invalid number of features.")
        if next_state.numel() != self.config.state_dim:
            raise ValueError("Next state has an invalid number of features.")

        self.replay_buffer.append(
            Transition(
                state.detach().float().cpu().reshape(-1),
                action,
                float(reward),
                next_state.detach().float().cpu().reshape(-1),
                done,
            )
        )

    def learn(self) -> Optional[float]:
        """Perform one DQN update from replay memory.

        Returns:
            Mean squared Bellman loss, or `None` when memory is insufficient.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch: List[Transition] = random.sample(
            list(self.replay_buffer),
            self.config.batch_size,
        )
        states: torch.Tensor = torch.stack([item.state for item in batch]).to(
            self.device
        )
        actions: torch.Tensor = torch.tensor(
            [item.action for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards: torch.Tensor = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states: torch.Tensor = torch.stack(
            [item.next_state for item in batch]
        ).to(self.device)
        dones: torch.Tensor = torch.tensor(
            [item.done for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        current_q: torch.Tensor = self.online_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q: torch.Tensor = self.target_network(next_states).max(dim=1).values
            target_q: torch.Tensor = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss: torch.Tensor = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

        if self.steps % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        return float(loss.detach().cpu().item())

    @staticmethod
    def compute_reward(
        dice_delta: float,
        auroc_delta: float,
        loss_delta: float,
    ) -> float:
        """Compute a validation-based reward.

        Args:
            dice_delta: Change in validation Dice.
            auroc_delta: Change in validation AUROC.
            loss_delta: Change in validation loss; positive means degradation.

        Returns:
            Scalar reward.
        """
        return float(dice_delta + 0.5 * auroc_delta - 0.1 * max(loss_delta, 0.0))

    def get_action_name(self, action: int) -> str:
        """Return the string identifier for an action index.

        Args:
            action: Action integer index.

        Returns:
            Semantic action name string.

        Raises:
            ValueError: If the action index is out of bounds.
        """
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError(f"Action index {action} out of bounds for {len(self.ACTIONS)} actions.")
        return self.ACTIONS[action]

    @staticmethod
    def construct_state_vector(
        train_loss: float,
        val_metric: float,
        loss_slope: float,
        progress: float,
        policy_factor: float,
    ) -> torch.Tensor:
        """Construct normalized 5-dimensional state vector for the Q-network.

        Args:
            train_loss: Current training loss (will be normalized via tanh).
            val_metric: Current validation metric (e.g. Dice or ROC-AUC in [0, 1]).
            loss_slope: Recent change in loss normalized to [-1.0, 1.0].
            progress: Epoch or step progress in [0.0, 1.0].
            policy_factor: Current dynamic augmentation or boundary factor in [0.0, 1.0].

        Returns:
            torch.Tensor: Normalized tensor shaped `(5,)` of dtype float32.
        """
        norm_loss = float(torch.tanh(torch.tensor(train_loss, dtype=torch.float32)).item())
        clamped_metric = float(max(0.0, min(1.0, val_metric)))
        clamped_slope = float(max(-1.0, min(1.0, loss_slope)))
        clamped_progress = float(max(0.0, min(1.0, progress)))
        clamped_factor = float(max(0.0, min(1.0, policy_factor)))

        return torch.tensor(
            [norm_loss, clamped_metric, clamped_slope, clamped_progress, clamped_factor],
            dtype=torch.float32,
        )

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save Q-network weights, optimizer state, and exploration parameters.

        Args:
            checkpoint_path: Destination path for .pt checkpoint file.
            metadata: Optional additional metadata dictionary to store.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "online_network_state_dict": self.online_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": float(self.epsilon),
            "steps": int(self.steps),
            "config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
                "hidden_dim": self.config.hidden_dim,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay": self.config.epsilon_decay,
                "replay_capacity": self.config.replay_capacity,
                "batch_size": self.config.batch_size,
                "target_update_frequency": self.config.target_update_frequency,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, str(path))
        logger.info("Saved DQN agent checkpoint to %s (steps=%d, epsilon=%.4f)", path, self.steps, self.epsilon)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load Q-network weights and parameters from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.

        Returns:
            Dict[str, Any]: Loaded checkpoint metadata dictionary.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"DQN Checkpoint not found: {path}")

        try:
            payload = torch.load(str(path), map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(str(path), map_location=self.device)

        self.online_network.load_state_dict(payload["online_network_state_dict"])
        self.target_network.load_state_dict(payload["target_network_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.steps = int(payload.get("steps", self.steps))
        logger.info("Loaded DQN agent checkpoint from %s (steps=%d, epsilon=%.4f)", path, self.steps, self.epsilon)
        return payload.get("metadata", {})