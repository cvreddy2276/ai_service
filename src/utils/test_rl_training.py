"""Test and verification script for Reinforcement Learning training pipelines.

Validates:
1. DQNPatchSamplingAgent network architecture, action selection, replay buffer, and Bellman update.
2. RLTrainingEnvironment state management and reward calculations.
3. RLSegmentationTrainer 2-epoch dry run on 3D DCE-MRI subtraction volumes.
4. RLClassificationTrainer 2-epoch dry run on 3D volumetric DCE-MRI tensors.
5. Generation of diagnostic verification plots documenting RL actions, rewards, and Q-values.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rl.environment import ActionInterpreter, RLTrainingEnvironment, SamplingActionPolicy
from src.rl.patch_sampling_agent import AgentConfig, DQNPatchSamplingAgent
from src.rl.rl_classification_trainer import RLClassificationTrainer
from src.rl.rl_segmentation_trainer import RLSegmentationTrainer
from src.utils.test_classification import prepare_synthetic_3d_dataset
from src.utils.test_segmentation import prepare_synthetic_segmentation_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger("TestRLTraining")

REPORTS_DIR: Path = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR: Path = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def test_dqn_agent_standalone(device: torch.device) -> None:
    """Validate DQNPatchSamplingAgent unit mechanics."""
    logger.info("--- Testing DQNPatchSamplingAgent Mechanics ---")
    config = AgentConfig(
        state_dim=5,
        action_dim=5,
        hidden_dim=64,
        batch_size=4,
        replay_capacity=100,
        target_update_frequency=10,
    )
    agent = DQNPatchSamplingAgent(config=config, device=device)

    # 1. State vector construction & shape
    state = DQNPatchSamplingAgent.construct_state_vector(
        train_loss=0.45,
        val_metric=0.72,
        loss_slope=-0.05,
        progress=0.3,
        policy_factor=0.5,
    )
    assert state.shape == (5,), f"Expected shape (5,), got {state.shape}"
    assert state.dtype == torch.float32

    # 2. Action selection
    action = agent.select_action(state, training=True)
    assert 0 <= action < 5, f"Selected action {action} out of bounds"
    action_name = agent.get_action_name(action)
    logger.info("Selected Action: %d (%s) from state %s", action, action_name, state.tolist())

    # 3. Replay buffer and learning step
    for i in range(10):
        s = DQNPatchSamplingAgent.construct_state_vector(0.5 - i * 0.02, 0.6 + i * 0.03, -0.02, i / 10.0, 0.5)
        next_s = DQNPatchSamplingAgent.construct_state_vector(0.48 - i * 0.02, 0.63 + i * 0.03, -0.02, (i + 1) / 10.0, 0.5)
        r = DQNPatchSamplingAgent.compute_reward(dice_delta=0.03, auroc_delta=0.0, loss_delta=-0.02)
        agent.remember(s, action=i % 5, reward=r, next_state=next_s, done=(i == 9))

    loss_q = agent.learn()
    assert loss_q is not None, "DQN learn() should return float loss when memory >= batch_size"
    logger.info("DQN Bellman Update Loss: %.4f | Steps: %d | Epsilon: %.4f", loss_q, agent.steps, agent.epsilon)

    # 4. Checkpoint save & load
    test_ckpt = CHECKPOINTS_DIR / "test_dqn_agent.pt"
    agent.save_checkpoint(test_ckpt, metadata={"test": True})
    assert test_ckpt.exists()

    new_agent = DQNPatchSamplingAgent(config=config, device=device)
    meta = new_agent.load_checkpoint(test_ckpt)
    assert meta.get("test") is True
    logger.info("DQN Agent Checkpoint Save & Load: PASSED")


def test_rl_environment_standalone() -> None:
    """Validate RLTrainingEnvironment transitions and reward logic."""
    logger.info("--- Testing RLTrainingEnvironment ---")
    env = RLTrainingEnvironment(task="segmentation", max_epochs=5)
    init_state = env.reset()
    assert init_state.shape == (5,)

    # Step 1
    next_s, r, done, info = env.step(action=0, train_loss=0.50, val_metric=0.60, epoch=1)
    assert not done
    assert "policy" in info
    logger.info("Env Step 1: Action=%s, Reward=%.4f", info["action_name"], r)

    # Step 2 with improvement
    next_s2, r2, done2, info2 = env.step(action=1, train_loss=0.40, val_metric=0.75, epoch=2)
    assert r2 > 0.0, f"Expected positive reward on metric improvement, got {r2}"
    logger.info("Env Step 2 (Improved): Action=%s, Reward=%.4f (Dice Delta: %.4f)", info2["action_name"], r2, info2["metric_delta"])
    logger.info("RLTrainingEnvironment Mechanics: PASSED")


def test_rl_segmentation_pipeline(device: torch.device) -> Dict[str, Any]:
    """Execute 2-epoch dry run of RLSegmentationTrainer on synthetic data."""
    logger.info("--- Testing RL-Guided 3D UNet Segmentation Pipeline ---")
    data_dir = Path("data/segmentation_dryrun")
    dataset_samples = prepare_synthetic_segmentation_dataset(data_dir, num_cases=6)

    train_data = dataset_samples[:4]
    val_data = dataset_samples[4:]

    trainer = RLSegmentationTrainer(
        train_data=train_data,
        val_data=val_data,
        spatial_size=(96, 96, 32),
        batch_size=2,
        num_samples=2,
        checkpoint_dir=CHECKPOINTS_DIR,
        checkpoint_name="test_best_rl_monai_3dunet.pt",
        agent_checkpoint_name="test_best_rl_segmentation_agent.pt",
        device=device,
        seed=42,
    )

    results = trainer.fit(max_epochs=2, val_interval=1)

    assert len(results["history"]["train_loss"]) == 2
    assert len(results["history"]["val_dice"]) == 2
    assert len(results["history"]["rl_action"]) == 2
    assert len(results["history"]["rl_reward"]) == 2
    assert Path(results["model_checkpoint_path"]).exists()
    assert Path(results["agent_checkpoint_path"]).exists()

    logger.info(
        "RL Segmentation 2-Epoch Dry Run Complete. Best Dice: %.4f, Actions: %s, Rewards: %s",
        results["best_val_dice"],
        results["history"]["rl_action_name"],
        results["history"]["rl_reward"],
    )
    return results


def test_rl_classification_pipeline(device: torch.device) -> Dict[str, Any]:
    """Execute 2-epoch dry run of RLClassificationTrainer on synthetic data."""
    logger.info("--- Testing RL-Guided 3D DCE-MRI Classification Pipeline ---")
    train_loader, val_loader = prepare_synthetic_3d_dataset(
        num_samples=16,
        spatial_size=(64, 64, 24),
        seed=42,
    )

    trainer = RLClassificationTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        backbone="simple_cnn3d",
        in_channels=1,
        num_classes=1,
        batch_size=4,
        checkpoint_dir=CHECKPOINTS_DIR,
        checkpoint_name="test_best_rl_3d_classifier.pt",
        agent_checkpoint_name="test_best_rl_classification_agent.pt",
        device=device,
        seed=42,
    )

    results = trainer.fit(epochs=2)

    assert len(results["history"]["train_loss"]) == 2
    assert len(results["history"]["val_auc"]) == 2
    assert len(results["history"]["rl_action"]) == 2
    assert len(results["history"]["rl_reward"]) == 2
    assert Path(results["model_checkpoint_path"]).exists()
    assert Path(results["agent_checkpoint_path"]).exists()

    logger.info(
        "RL Classification 2-Epoch Dry Run Complete. Best AUC: %.4f, Actions: %s, Rewards: %s",
        results["best_val_auc"],
        results["history"]["rl_action_name"],
        results["history"]["rl_reward"],
    )
    return results


def generate_rl_diagnostic_plots(
    seg_results: Dict[str, Any],
    cls_results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate diagnostic figure illustrating RL co-training dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#161b22")

    for ax in axes.flat:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["top"].set_color("#30363d")
        ax.spines["right"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    epochs = [1, 2]

    # Panel 1: Loss curves
    axes[0, 0].plot(epochs, seg_results["history"]["train_loss"], marker="o", color="#58a6ff", label="Segmentation Loss")
    axes[0, 0].plot(epochs, cls_results["history"]["train_loss"], marker="s", color="#f0883e", label="Classification Loss")
    axes[0, 0].set_title("Training Loss Trajectory", fontsize=12, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    axes[0, 0].grid(True, linestyle="--", alpha=0.2, color="white")

    # Panel 2: Validation Metrics
    axes[0, 1].plot(epochs, seg_results["history"]["val_dice"], marker="^", color="#3fb950", label="Val Dice (Segmentation)")
    axes[0, 1].plot(epochs, cls_results["history"]["val_auc"], marker="d", color="#bc8cff", label="Val ROC-AUC (Classification)")
    axes[0, 1].set_title("Validation Metric Performance", fontsize=12, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score [0.0 - 1.0]")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    axes[0, 1].grid(True, linestyle="--", alpha=0.2, color="white")

    # Panel 3: RL Rewards
    axes[1, 0].plot(epochs, seg_results["history"]["rl_reward"], marker="o", color="#38bdf8", label="Seg Agent Reward")
    axes[1, 0].plot(epochs, cls_results["history"]["rl_reward"], marker="s", color="#f43f5e", label="Cls Agent Reward")
    axes[1, 0].axhline(0.0, color="#8b949e", linestyle=":", alpha=0.6)
    axes[1, 0].set_title("RL Validation Reward Signal", fontsize=12, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    axes[1, 0].grid(True, linestyle="--", alpha=0.2, color="white")

    # Panel 4: Actions Chosen
    all_actions = seg_results["history"]["rl_action_name"] + cls_results["history"]["rl_action_name"]
    unique_actions, counts = np.unique(all_actions, return_counts=True)
    axes[1, 1].bar(unique_actions, counts, color="#79c0ff", edgecolor="#1f6feb", alpha=0.85)
    axes[1, 1].set_title("RL Policy Action Frequency", fontsize=12, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("Selected Action")
    axes[1, 1].set_ylabel("Frequency Count")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(True, linestyle="--", alpha=0.2, color="white", axis="y")

    plt.suptitle(
        "Reinforcement Learning Co-Training Verification (3D DCE-MRI Pipeline)",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved RL verification diagnostic plot to %s", output_path)
    return output_path


def run_all_rl_tests() -> None:
    """Execute end-to-end verification of the RL components."""
    logger.info("================================================================")
    logger.info("Starting Reinforcement Learning Co-Training Pipeline Verification")
    logger.info("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Compute Device: %s", device)

    # 1. DQN Agent Mechanics
    test_dqn_agent_standalone(device)

    # 2. Environment Mechanics
    test_rl_environment_standalone()

    # 3. RL-Guided 3D UNet Segmentation Dry Run
    seg_results = test_rl_segmentation_pipeline(device)

    # 4. RL-Guided 3D Classification Dry Run
    cls_results = test_rl_classification_pipeline(device)

    # 5. Diagnostic Plot Export
    plot_path = REPORTS_DIR / "rl_training_diagnostic.png"
    generate_rl_diagnostic_plots(seg_results, cls_results, plot_path)

    logger.info("================================================================")
    logger.info("Reinforcement Learning Co-Training Pipeline Verification: PASSED!")
    logger.info("================================================================")


if __name__ == "__main__":
    run_all_rl_tests()
