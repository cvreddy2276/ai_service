"""3D Deep Learning Model Evaluation, Metrics Computation, and Diagnostic Plot Generation.

Computes ROC-AUC, Sensitivity, Specificity, Precision, F1-Score, and Confusion
Matrix from 3D Deep Learning model predictions and plots training curves and ROC curves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification performance metrics.

    Args:
        y_true: Ground truth binary labels array of shape (N,).
        y_prob: Predicted positive-class probabilities of shape (N,).
        threshold: Decision threshold for binary prediction (default: 0.5).

    Returns:
        Dict[str, float]: Dictionary of metric names to scalar float values.
    """
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be one-dimensional arrays with equal length.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute classification metrics for an empty cohort.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must contain finite probabilities in [0, 1].")
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC-AUC requires both classes in the evaluation cohort.")
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn + 1e-7))   # Recall / True Positive Rate
    specificity = float(tn / (tn + fp + 1e-7))   # True Negative Rate
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    balanced_accuracy = (sensitivity + specificity) / 2.0

    metrics = {
        "ROC_AUC": roc_auc,
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Brier_Score": float(brier_score_loss(y_true, y_prob)),
        "Sensitivity_Recall": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1_Score": f1,
        "Balanced_Accuracy": balanced_accuracy,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }
    return metrics


def plot_single_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    model_label: str = "3D Deep Learning Classifier",
) -> None:
    """Plot ROC curve for a 3D Deep Learning model evaluation.

    Args:
        y_true: Array of true binary labels (0 or 1).
        y_prob: Array of predicted probabilities [0.0 - 1.0].
        output_path: Path to save the PNG figure.
        model_label: Display label for the model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be one-dimensional arrays with equal length.")
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("ROC plotting requires a nonempty cohort containing both classes.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must contain finite probabilities in [0, 1].")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1a73e8", lw=2.5, label=f"{model_label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance Level")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"3D Deep Learning ROC Curve — {model_label}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve plot to %s", output_path)


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "3D Deep Learning Classifier Training History",
) -> None:
    """Plot training and validation loss and metric convergence curves over epochs.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'val_auc', etc.
        output_path: Destination path for the saved PNG figure.
        title: Overall figure super-title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Loss curves
    axes[0].plot(epochs, history.get("train_loss", []), "o-", color="#d93025", label="Train Loss", lw=2)
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], "s--", color="#1a73e8", label="Val Loss", lw=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (BCEWithLogits)", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # 2. Validation AUC & F1 curves
    if "val_auc" in history and history["val_auc"]:
        axes[1].plot(epochs, history["val_auc"], "^-", color="#1e8e3e", label="Val ROC-AUC", lw=2)
    if "val_f1" in history and history["val_f1"]:
        axes[1].plot(epochs, history["val_f1"], "d--", color="#f9ab00", label="Val F1-Score", lw=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Score [0 - 1]", fontsize=12)
    axes[1].set_title("Validation Discriminative Metrics", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves plot to %s", output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    class_names: Sequence[str] = ("non-pCR", "pCR"),
    title: str = "3D Deep Learning Confusion Matrix",
) -> None:
    """Plot annotated confusion matrix heatmap.

    Args:
        y_true: Ground truth binary labels (N,).
        y_pred: Predicted binary labels (N,).
        output_path: Absolute path to save the figure (PNG).
        class_names: Class label names for axis annotations.
        title: Figure title string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-7)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", color=color, fontsize=12, fontweight="bold",
            )

    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", output_path)


def save_classification_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Save classification report as JSON.

    Args:
        model_results: Mapping of model name to metric dictionary.
        output_path: Path to save the JSON classification report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable: Dict[str, Any] = {}
    for model_name, metrics in model_results.items():
        serializable[model_name] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in metrics.items()
        }

    with open(str(output_path), "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved classification report to %s", output_path)
