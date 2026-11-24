"""Patient-grouped cross-validation for pCR classification research."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.classification.evaluate import compute_classification_metrics
from src.classification.trainer import DCE3DClassificationTrainer
from src.preprocessing.dataset import BreastDCEDataset, get_dce_dataloader

logger = logging.getLogger(__name__)


def _stratified_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
    repetitions: int = 2000,
) -> Dict[str, List[float]]:
    """Compute stratified percentile intervals for ROC-AUC and PR-AUC.

    Args:
        labels: Binary ground-truth labels with shape (N,).
        probabilities: Positive-class probabilities with shape (N,).
        seed: Random seed for reproducible resampling.
        repetitions: Number of bootstrap samples.

    Returns:
        Dictionary mapping metric names to [2.5th, 97.5th] percentile intervals.
    """
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        return {"ROC_AUC_95_CI": [float("nan"), float("nan")], "PR_AUC_95_CI": [float("nan"), float("nan")]}

    roc_values: List[float] = []
    pr_values: List[float] = []
    for _ in range(repetitions):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in class_indices
        ])
        metrics = compute_classification_metrics(labels[sampled], probabilities[sampled])
        roc_values.append(metrics["ROC_AUC"])
        pr_values.append(metrics["PR_AUC"])

    return {
        "ROC_AUC_95_CI": [float(x) for x in np.quantile(roc_values, [0.025, 0.975])],
        "PR_AUC_95_CI": [float(x) for x in np.quantile(pr_values, [0.025, 0.975])],
    }


def run_grouped_cross_validation(
    data_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
    seed: int = 42,
    n_splits: int = 5,
    max_epochs: int = 20,
    patience: int = 5,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Run stratified patient-grouped outer CV with a separate inner validation set.

    Every eligible patient contributes one out-of-fold estimate. The outer test fold
    is not used to select a checkpoint; an inner validation partition selects the
    best epoch for each fold.

    Args:
        data_dir: Root directory containing matched DCE and mask NIfTI files.
        metadata_csv: CSV containing patient IDs and binary pCR labels.
        output_dir: Destination for fold checkpoints and evaluation reports.
        seed: Base random seed for folds, training, and bootstrap intervals.
        n_splits: Number of outer patient-grouped folds.
        max_epochs: Maximum training epochs per fold.
        patience: Early-stopping patience based on validation ROC-AUC.
        batch_size: Number of volumes per training batch.

    Returns:
        Report dictionary containing fold-level and pooled out-of-fold metrics.

    Raises:
        ValueError: If cohort labels or parameters cannot support stratified folds.
        RuntimeError: If fold test partitions fail to cover every patient exactly once.
    """
    if n_splits < 2 or max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("n_splits >= 2, max_epochs >= 1, patience >= 1, and batch_size >= 1 are required.")

    set_determinism(seed=seed)
    torch.manual_seed(seed)
    metadata = pd.read_csv(metadata_csv)
    if "pid" not in metadata.columns or "pCR" not in metadata.columns:
        raise ValueError("Metadata CSV must contain 'pid' and 'pCR' columns.")
    metadata = metadata.dropna(subset=["pid", "pCR"]).copy()
    metadata["pid"] = metadata["pid"].astype(str)
    if metadata["pid"].duplicated().any():
        raise ValueError("Metadata contains duplicate patient IDs; resolve duplicates before evaluation.")
    if not metadata["pCR"].isin([0, 1]).all():
        raise ValueError("pCR labels must contain only binary values 0 and 1.")

    discovered = BreastDCEDataset(data_dir, metadata_csv=metadata_csv, transform=None)
    available = pd.DataFrame([
        {"pid": str(sample["pid"]), "label": int(sample["label"].item())}
        for sample in discovered.samples
    ])
    if available["pid"].duplicated().any():
        raise ValueError("Multiple volumes per patient require a patient-level grouping rule before CV.")
    class_counts = available["label"].value_counts()
    if len(available) < n_splits * 2 or len(class_counts) != 2 or class_counts.min() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} patients in each class for {n_splits}-fold CV; "
            f"found {class_counts.to_dict()}."
        )

    labels_by_pid = available.set_index("pid")["label"]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records: List[Dict[str, Any]] = []
    oof_records: List[Dict[str, Any]] = []
    all_ids = set(available["pid"].tolist())
    tested_ids: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_number, (outer_train_idx, outer_test_idx) in enumerate(
        splitter.split(available["pid"], available["label"], groups=available["pid"]),
        start=1,
    ):
        outer_train_ids = available.iloc[outer_train_idx]["pid"].tolist()
        # The dataset loader discovers volumes in sorted filename order; sort IDs
        # so the returned test predictions stay paired with their patient IDs.
        test_ids = sorted(available.iloc[outer_test_idx]["pid"].tolist())
        test_labels = labels_by_pid.loc[test_ids].to_numpy(dtype=np.int64)
        if len(np.unique(test_labels)) != 2:
            raise ValueError(f"Outer test fold {fold_number} does not contain both pCR classes.")

        inner_train_ids, validation_ids = train_test_split(
            outer_train_ids,
            test_size=0.1875,
            random_state=seed + fold_number,
            stratify=labels_by_pid.loc[outer_train_ids].to_numpy(dtype=np.int64),
        )
        inner_train_ids = [str(pid) for pid in inner_train_ids]
        validation_ids = [str(pid) for pid in validation_ids]
        if set(inner_train_ids) & set(validation_ids) or set(inner_train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
            raise RuntimeError(f"Patient leakage detected in fold {fold_number}.")

        train_labels = labels_by_pid.loc[inner_train_ids]
        positive_weight = float((train_labels == 0).sum() / max(int((train_labels == 1).sum()), 1))
        train_loader = get_dce_dataloader(
            data_dir, metadata_csv, patient_ids=inner_train_ids, batch_size=batch_size,
            shuffle=True, is_training=True, use_cache=False,
        )
        validation_loader = get_dce_dataloader(
            data_dir, metadata_csv, patient_ids=validation_ids, batch_size=1,
            shuffle=False, is_training=False, use_cache=False,
        )
        test_loader = get_dce_dataloader(
            data_dir, metadata_csv, patient_ids=test_ids, batch_size=1,
            shuffle=False, is_training=False, use_cache=False,
        )

        fold_seed = seed + fold_number
        set_determinism(seed=fold_seed)
        torch.manual_seed(fold_seed)
        checkpoint_path = output_dir / f"fold_{fold_number}_best_3d_classifier.pt"
        trainer = DCE3DClassificationTrainer(
            backbone="simple_cnn3d", in_channels=1, num_classes=1,
            pos_weight=positive_weight, device=None,
        )
        trainer.fit(
            train_loader,
            validation_loader,
            epochs=max_epochs,
            checkpoint_path=checkpoint_path,
            early_stopping_patience=patience,
        )
        selected_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        selected_epoch = int(selected_checkpoint.get("epoch", max_epochs))
        trainer.load_checkpoint(checkpoint_path)
        _, _, fold_labels, fold_probabilities = trainer.validate(test_loader)
        fold_metrics = compute_classification_metrics(fold_labels, fold_probabilities)

        fold_records.append({
            "fold": fold_number,
            "train_size": len(inner_train_ids),
            "validation_size": len(validation_ids),
            "test_size": len(test_ids),
            "test_class_counts": {str(label): int((test_labels == label).sum()) for label in (0, 1)},
            "selected_epoch": selected_epoch,
            "metrics": fold_metrics,
        })
        tested_ids.extend(test_ids)
        oof_records.extend([
            {"pid": pid, "label": int(label), "pcr_probability": float(probability), "fold": fold_number}
            for pid, label, probability in zip(test_ids, fold_labels, fold_probabilities)
        ])
        logger.info("Completed outer fold %d/%d: ROC-AUC %.4f", fold_number, n_splits, fold_metrics["ROC_AUC"])

        del trainer, train_loader, validation_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(tested_ids) != len(all_ids) or set(tested_ids) != all_ids or len(set(tested_ids)) != len(tested_ids):
        raise RuntimeError("Outer folds did not evaluate every matched patient exactly once.")

    oof_frame = pd.DataFrame(oof_records).sort_values("pid").reset_index(drop=True)
    labels = oof_frame["label"].to_numpy(dtype=np.int64)
    probabilities = oof_frame["pcr_probability"].to_numpy(dtype=np.float32)
    pooled_metrics = compute_classification_metrics(labels, probabilities)
    intervals = _stratified_bootstrap_intervals(labels, probabilities, seed=seed)
    fold_metric_names = ["ROC_AUC", "PR_AUC", "Brier_Score", "Balanced_Accuracy", "Sensitivity_Recall", "Specificity"]
    fold_summary = {
        name: {
            "mean": float(np.mean([fold["metrics"][name] for fold in fold_records])),
            "std": float(np.std([fold["metrics"][name] for fold in fold_records], ddof=1)),
        }
        for name in fold_metric_names
    }
    source_values = sorted(metadata["dataset"].dropna().astype(str).unique().tolist()) if "dataset" in metadata else []
    report: Dict[str, Any] = {
        "task": "pCR response prediction",
        "evaluation_design": "StratifiedGroupKFold outer evaluation with a separate patient-disjoint inner validation split for checkpoint selection.",
        "seed": seed,
        "n_splits": n_splits,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "cohort_size": len(available),
        "cohort_sources": source_values,
        "class_counts": {str(int(key)): int(value) for key, value in class_counts.items()},
        "threshold": 0.5,
        "folds": fold_records,
        "fold_metric_summary": fold_summary,
        "pooled_out_of_fold_metrics": pooled_metrics,
        "pooled_out_of_fold_confidence_intervals": intervals,
        "external_validation": "Not performed: no independent external cohort is present in this workspace.",
        "interpretation": "Internal cross-validation is exploratory. It does not establish performance on a new site, scanner, protocol, or real-world clinical population.",
    }
    oof_frame.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    (output_dir / "cross_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved cross-validation report to %s", output_dir / "cross_validation_report.json")
    return report


def main() -> None:
    """Parse command-line options and run patient-grouped cross-validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("BreastDCEDL_spy1"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation/pcr_grouped_5fold"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run_grouped_cross_validation(
        data_dir=args.data_dir,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        n_splits=args.n_splits,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    print(json.dumps(report["pooled_out_of_fold_metrics"], indent=2))


if __name__ == "__main__":
    main()
