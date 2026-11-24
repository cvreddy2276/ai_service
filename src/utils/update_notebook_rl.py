"""Utility script to update breast_dce_dl_pipeline.ipynb with executable RL training cells.

Updates:
1. Fixes Cell 1 imports so that sys.path includes root_dir before importing src.rl.
2. Imports all reinforcement learning components:
   DQNPatchSamplingAgent, AgentConfig, RLSegmentationTrainer, RLClassificationTrainer, RLTrainingEnvironment.
3. Removes empty/stale cells.
4. Adds Section 7: Reinforcement Learning (DQN) Active Curriculum & Patch Sampling Training
   with interactive agent decision-making, 3D co-training demonstration on real cases,
   and diagnostic visualization plotting.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UpdateNotebookRL")


def update_notebook() -> None:
    """Read, update, and save the Jupyter notebook."""
    nb_path = Path("notebooks/breast_dce_dl_pipeline.ipynb")
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found at: {nb_path}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    logger.info("Loaded notebook with %d cells", len(nb["cells"]))

    # 1. Update Cell 1 (Imports)
    cell_1_source = [
        "import sys\n",
        "import logging\n",
        "from pathlib import Path\n",
        "\n",
        "# Add project root to sys.path first to support both root and notebooks directory execution\n",
        "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
        "if str(root_dir) not in sys.path:\n",
        "    sys.path.insert(0, str(root_dir))\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import SimpleITK as sitk\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "from PIL import Image\n",
        "\n",
        "# Import Reinforcement Learning Co-Training components\n",
        "from src.rl import (\n",
        "    AgentConfig,\n",
        "    DQNPatchSamplingAgent,\n",
        "    RLTrainingEnvironment,\n",
        "    RLSegmentationTrainer,\n",
        "    RLClassificationTrainer,\n",
        ")\n",
        "\n",
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')\n",
        "logger = logging.getLogger('BreastDCEDL_Pipeline')\n",
        "logger.info('Initialized 3D DCE-MRI Deep Learning Pipeline with RL Co-Training on real data')"
    ]
    nb["cells"][1]["source"] = cell_1_source

    # 2. Filter out empty cells
    cleaned_cells = [c for c in nb["cells"] if not (c["cell_type"] == "code" and not "".join(c.get("source", [])).strip())]
    logger.info("Retained %d cells after removing empty cells", len(cleaned_cells))

    # 3. Check if Section 7 is already present
    has_section_7 = any(
        "## 7. Reinforcement Learning" in "".join(c.get("source", []))
        for c in cleaned_cells
    )

    if not has_section_7:
        # Markdown Cell for Section 7
        section_7_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Reinforcement Learning (DQN) Active Curriculum & Patch Sampling Training\n",
                "\n",
                "In this section, we run and evaluate the **Reinforcement Learning Co-Training Engine** (`src/rl/`):\n",
                "- The **`DQNPatchSamplingAgent`** observes normalized 5D training state vectors:\n",
                "  $$\\mathbf{s} = [\\tanh(\\mathcal{L}), \\text{Metric}, \\Delta \\mathcal{L}, \\text{Progress}, \\text{PolicyFactor}]$$\n",
                "- The agent adaptively selects from 5 medical sampling & curriculum actions:\n",
                "  - `0: tumor_center_patch` — High foreground weight on tumor core (`pos=1.0, neg=0.05`)\n",
                "  - `1: lesion_boundary_patch` — Balanced focus on tumor-parenchyma margin (`pos=0.6, neg=0.4`)\n",
                "  - `2: background_patch` — Hard negative mining of healthy parenchyma and chest wall (`pos=0.05, neg=0.95`)\n",
                "  - `3: increase_augmentation` — Increases 3D rotation, flip, and noise perturbation to prevent overfitting\n",
                "  - `4: decrease_augmentation` — Reduces augmentations to stabilize gradient descent\n",
                "- Experience transitions $(\\mathbf{s}_t, a_t, r_t, \\mathbf{s}_{t+1}, \\text{done})$ are stored in replay memory and optimized via smooth L1 Bellman loss."
            ]
        }

        # Code Cell: RL Agent Inspection & Interactive Step
        section_7_code_1 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7.1 Inspect DQNPatchSamplingAgent and Action Space\n",
                "rl_config = AgentConfig(\n",
                "    state_dim=5,\n",
                "    action_dim=5,\n",
                "    hidden_dim=128,\n",
                "    learning_rate=1e-4,\n",
                "    batch_size=4,\n",
                "    replay_capacity=1000,\n",
                ")\n",
                "rl_agent = DQNPatchSamplingAgent(config=rl_config, device=device)\n",
                "\n",
                "logger.info('Instantiated DQN Patch Sampling Agent with %d actions:', len(rl_agent.ACTIONS))\n",
                "for idx, act in enumerate(rl_agent.ACTIONS):\n",
                "    print(f'  Action {idx}: {act}')\n",
                "\n",
                "# Construct state from current model loss & metrics\n",
                "sample_state = DQNPatchSamplingAgent.construct_state_vector(\n",
                "    train_loss=0.42,\n",
                "    val_metric=0.75,\n",
                "    loss_slope=-0.03,\n",
                "    progress=0.5,\n",
                "    policy_factor=0.5,\n",
                ")\n",
                "selected_action = rl_agent.select_action(sample_state, training=True)\n",
                "action_name = rl_agent.get_action_name(selected_action)\n",
                "\n",
                "print('\\n--- SAMPLE RL AGENT DECISION ---')\n",
                "print(f'State Vector (5D)     : {sample_state.tolist()}')\n",
                "print(f'Selected Policy Action: [{selected_action}] {action_name}')\n",
                "print(f'Current Exploration ε : {rl_agent.epsilon:.4f}')"
            ]
        }

        # Code Cell: Running RL-Guided Co-Training Demonstration
        section_7_code_2 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7.2 Execute RL-Guided 3D Deep Learning Co-Training Demonstration on Real Cases\n",
                "logger.info('Executing RL-Guided 3D Classification Training demonstration...')\n",
                "\n",
                "# Prepare train/validation sample dictionaries from real I-SPY1 cases discovered in Section 1\n",
                "real_train_samples = []\n",
                "real_val_samples = []\n",
                "\n",
                "for i, rec in enumerate(patient_records[:8]):\n",
                "    sample_item = {\n",
                "        'pid': rec['pid'],\n",
                "        't0': rec['t0_path'],\n",
                "        't1': rec['t1_path'],\n",
                "        'mask': rec['mask_path'],\n",
                "        'label': torch.tensor(float(rec['pCR']), dtype=torch.float32),\n",
                "    }\n",
                "    if i < 6:\n",
                "        real_train_samples.append(sample_item)\n",
                "    else:\n",
                "        real_val_samples.append(sample_item)\n",
                "\n",
                "# Instantiate RL-Guided 3D Classification Trainer\n",
                "rl_trainer = RLClassificationTrainer(\n",
                "    train_samples=real_train_samples,\n",
                "    val_samples=real_val_samples,\n",
                "    backbone='simple_cnn3d',\n",
                "    in_channels=1,\n",
                "    num_classes=1,\n",
                "    batch_size=2,\n",
                "    base_roi_size=(64, 64, 24),\n",
                "    checkpoint_dir=root_dir / 'checkpoints',\n",
                "    checkpoint_name='demo_rl_classifier.pt',\n",
                "    agent_checkpoint_name='demo_rl_agent.pt',\n",
                "    device=device,\n",
                "    seed=42,\n",
                ")\n",
                "\n",
                "# Run 2-epoch RL co-training pass\n",
                "rl_summary = rl_trainer.fit(epochs=2)\n",
                "\n",
                "print('\\n--- RL CO-TRAINING EXECUTION SUMMARY ---')\n",
                "print(f'Best Validation ROC-AUC : {rl_summary[\"best_val_auc\"]:.4f}')\n",
                "print(f'Actions Selected Per Ep : {rl_summary[\"history\"][\"rl_action_name\"]}')\n",
                "print(f'RL Rewards Earned       : {[round(r, 4) for r in rl_summary[\"history\"][\"rl_reward\"]]}')\n",
                "print(f'Model Checkpoint Path   : {rl_summary[\"model_checkpoint_path\"]}')"
            ]
        }

        # Code Cell: Diagnostic Plotting of RL Dynamics
        section_7_code_3 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7.3 Visualize RL Co-Training Dynamics\n",
                "fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))\n",
                "\n",
                "epochs = [1, 2]\n",
                "hist = rl_summary['history']\n",
                "\n",
                "# Panel 1: Loss & Validation AUC Trajectory\n",
                "axes[0].plot(epochs, hist['train_loss'], marker='o', color='#1a73e8', label='Train Loss')\n",
                "axes[0].plot(epochs, hist['val_auc'], marker='s', color='#34a853', label='Val ROC-AUC')\n",
                "axes[0].set_title('1. Loss & Validation AUC Trajectory', fontsize=11, fontweight='bold')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].set_ylabel('Value')\n",
                "axes[0].set_xticks(epochs)\n",
                "axes[0].legend()\n",
                "axes[0].grid(alpha=0.3)\n",
                "\n",
                "# Panel 2: RL Reward Signal\n",
                "axes[1].plot(epochs, hist['rl_reward'], marker='^', color='#e3742f', lw=2, label='RL Reward')\n",
                "axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)\n",
                "axes[1].set_title('2. RL Agent Reward Signal (ΔMetric - ΔLoss)', fontsize=11, fontweight='bold')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_ylabel('Reward')\n",
                "axes[1].set_xticks(epochs)\n",
                "axes[1].legend()\n",
                "axes[1].grid(alpha=0.3)\n",
                "\n",
                "# Panel 3: Actions Selected\n",
                "actions_taken = hist['rl_action_name']\n",
                "unique_acts, act_counts = np.unique(actions_taken, return_counts=True)\n",
                "bars = axes[2].bar(unique_acts, act_counts, color='#8ab4f8', edgecolor='#1a73e8')\n",
                "axes[2].set_title('3. RL Policy Actions Triggered', fontsize=11, fontweight='bold')\n",
                "axes[2].set_xlabel('Selected Action')\n",
                "axes[2].set_ylabel('Count')\n",
                "axes[2].set_yticks(range(0, max(act_counts) + 2))\n",
                "axes[2].grid(axis='y', alpha=0.3)\n",
                "\n",
                "plt.suptitle('Reinforcement Learning (DQN) Active Co-Training Dashboard', fontsize=13, fontweight='bold', y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }

        cleaned_cells.extend([section_7_md, section_7_code_1, section_7_code_2, section_7_code_3])
        logger.info("Appended Section 7 cells to notebook")

    nb["cells"] = cleaned_cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    logger.info("Successfully saved updated notebook (%d cells) to %s", len(nb["cells"]), nb_path)


if __name__ == "__main__":
    update_notebook()
