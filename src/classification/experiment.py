"""Patient-level training, validation, and held-out evaluation for pCR classification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from monai.utils import set_determinism

from src.classification.evaluate import compute_classification_metrics
from src.classification.trainer import DCE3DClassificationTrainer
from src.preprocessing.dataset import BreastDCEDataset, get_dce_dataloader

logger = logging.getLogger(__name__)


def run_patient_level_experiment(
    data_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Train on patient-disjoint splits and evaluate the test partition once.

    Args:
        data_dir: Root containing DCE and mask folders.
        metadata_csv: Clinical metadata CSV containing pid and pCR columns.
        output_dir: Directory for the validation-selected checkpoint and test report.
        seed: Deterministic split and training seed.
        epochs: Maximum training epochs.
        batch_size: Patient volumes per batch.

    Returns:
        A report containing split patient IDs, counts, and held-out metrics.

    Raises:
        ValueError: If there are too few patients or either split lacks a class.
    """
    set_determinism(seed=seed)
    torch.manual_seed(seed)
    metadata = pd.read_csv(metadata_csv)
    if "pid" not in metadata or "pCR" not in metadata:
        raise ValueError("Metadata CSV must contain 'pid' and 'pCR' columns.")
    metadata = metadata.dropna(subset=["pid", "pCR"]).copy()
    metadata["pid"] = metadata["pid"].astype(str)
    if metadata["pid"].duplicated().any():
        duplicates = metadata.loc[metadata["pid"].duplicated(), "pid"].tolist()
        raise ValueError(f"Duplicate patient IDs in metadata: {duplicates[:10]}")
    if not metadata["pCR"].isin([0, 1]).all():
        raise ValueError("pCR labels must be binary values 0 or 1.")

    discovered = BreastDCEDataset(data_dir, metadata_csv=metadata_csv, transform=None)
    available = pd.DataFrame(
        [{"pid": str(sample["pid"]), "label": int(sample["label"].item())} for sample in discovered.samples]
    )
    if available["pid"].duplicated().any():
        raise ValueError("More than one matched volume was found per patient; define a patient-level grouping rule.")
    if len(available) < 20 or available["label"].nunique() != 2:
        raise ValueError(f"Need at least 20 matched patients with both pCR classes; found {len(available)}.")

    train_val_ids, test_ids = train_test_split(
        available["pid"].to_numpy(), test_size=0.15, random_state=seed,
        stratify=available["label"].to_numpy(),
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=available.set_index("pid").loc[train_val_ids, "label"].to_numpy(),
    )
    train_ids = [str(pid) for pid in train_ids]
    val_ids = [str(pid) for pid in val_ids]
    test_ids = [str(pid) for pid in test_ids]
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise RuntimeError("Patient leakage detected across data splits.")

    train_labels = available.set_index("pid").loc[train_ids, "label"]
    pos_weight = float((train_labels == 0).sum() / max((train_labels == 1).sum(), 1))
    train_loader = get_dce_dataloader(
        data_dir, metadata_csv, patient_ids=train_ids, batch_size=batch_size,
        shuffle=True, is_training=True, use_cache=False,
    )
    val_loader = get_dce_dataloader(
        data_dir, metadata_csv, patient_ids=val_ids, batch_size=1,
        shuffle=False, is_training=False, use_cache=False,
    )
    test_loader = get_dce_dataloader(
        data_dir, metadata_csv, patient_ids=test_ids, batch_size=1,
        shuffle=False, is_training=False, use_cache=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_3d_classifier.pt"
    trainer = DCE3DClassificationTrainer(
        backbone="simple_cnn3d", in_channels=1, num_classes=1,
        pos_weight=pos_weight, device=None,
    )
    trainer.fit(train_loader, val_loader, epochs=epochs, checkpoint_path=checkpoint)
    trainer.load_checkpoint(checkpoint)
    _, _, y_true, y_prob = trainer.validate(test_loader)
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    report: Dict[str, Any] = {
        "task": "pCR response prediction",
        "seed": seed,
        "threshold": 0.5,
        "cohort_size": len(available),
        "class_counts": {str(int(k)): int(v) for k, v in available["label"].value_counts().items()},
        "split_patient_ids": {"train": train_ids, "validation": val_ids, "test": test_ids},
        "split_sizes": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "held_out_test_metrics": metrics,
        "checkpoint": str(checkpoint),
        "evaluation_note": "Single deterministic split; test metrics are exploratory and require external validation.",
    }
    report_path = output_dir / "held_out_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved held-out test report to %s", report_path)
    return report


def main() -> None:
    """Parse CLI arguments and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("BreastDCEDL_spy1"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/pcr_patient_split"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_patient_level_experiment(args.data_dir, args.metadata_csv, args.output_dir, args.seed, args.epochs, args.batch_size)
    print(json.dumps(report["held_out_test_metrics"], indent=2))


if __name__ == "__main__":
    main()
