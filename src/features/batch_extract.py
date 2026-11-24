"""Batch Radiomics Feature Extraction for Breast DCE-MRI Datasets.

Extracts radiomics features across patient cohorts, integrates clinical
metadata endpoints, and exports consolidated tabular CSV datasets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from tqdm import tqdm

from src.features.radiomics_extractor import DCEmriRadiomicsExtractor
from src.preprocessing.mri_processor import compute_subtraction_volume, resample_to_isotropic
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BatchRadiomics")


def extract_dataset_radiomics(
    data_dir: Union[str, Path],
    output_csv_path: Union[str, Path] = "data/processed/radiomics_features.csv",
    metadata_csv: Optional[Union[str, Path]] = None,
    bin_width: float = 25.0,
    max_cases: Optional[int] = None,
) -> pd.DataFrame:
    """Extract radiomics features across all cases in a dataset folder and export to CSV.

    Args:
        data_dir: Directory containing preprocessed subtraction/mask files or DCE/mask folders.
        output_csv_path: Destination path for exported radiomics CSV.
        metadata_csv: Optional path to clinical metadata CSV (e.g. BreastDCEDL_spy1_metadata.csv).
        bin_width: Discretization bin width for texture calculation.
        max_cases: Optional maximum number of cases to process.

    Returns:
        pd.DataFrame: Consolidated radiomics feature table with clinical metadata.
    """
    data_p = Path(data_dir)
    out_p = Path(output_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata dataframe if available
    metadata_df: Optional[pd.DataFrame] = None
    if metadata_csv and Path(metadata_csv).exists():
        metadata_df = pd.read_csv(metadata_csv)
        if "pid" in metadata_df.columns:
            metadata_df["pid"] = metadata_df["pid"].astype(str)
            metadata_df.set_index("pid", inplace=True)

    extractor = DCEmriRadiomicsExtractor(bin_width=bin_width)
    records: List[Dict[str, Any]] = []

    # 1. Discover subtraction and mask files
    # Check for direct subtraction folder (e.g. data/segmentation_dryrun/subtraction)
    sub_files = list(data_p.glob("**/subtraction/*_subtraction.nii*")) + list(data_p.glob("**/*_subtraction.nii*"))
    
    # If not pre-subtracted, discover paired T0/T1 and masks
    if len(sub_files) == 0:
        dce_dir = data_p / "spt1_dce" if (data_p / "spt1_dce").exists() else data_p
        mask_dir = data_p / "spy1_mask" if (data_p / "spy1_mask").exists() else data_p
        acq0_files = sorted(list(dce_dir.glob("*_acq0.nii*")) + list(dce_dir.glob("*_pre.nii*")))
    else:
        acq0_files = []

    logger.info("Found %d subtraction files and %d paired DCE files in %s", len(sub_files), len(acq0_files), data_p)

    # Process Pre-subtracted files
    if len(sub_files) > 0:
        if max_cases:
            sub_files = sub_files[:max_cases]

        for sub_p in tqdm(sub_files, desc="Extracting Radiomics"):
            case_id = sub_p.name.replace("_subtraction.nii.gz", "").replace("_subtraction.nii", "")
            mask_p = sub_p.parent.parent / "masks" / f"{case_id}_mask.nii.gz"
            if not mask_p.exists():
                mask_candidates = list(sub_p.parent.parent.glob(f"**/{case_id}*mask.nii*"))
                if mask_candidates:
                    mask_p = mask_candidates[0]
                else:
                    continue

            try:
                feats = extractor.extract_from_files(sub_p, mask_p)
                row: Dict[str, Any] = {"pid": case_id, **feats}

                # Attach metadata
                if metadata_df is not None and case_id in metadata_df.index:
                    meta_row = metadata_df.loc[case_id].to_dict()
                    for col, val in meta_row.items():
                        if col != "pid":
                            row[f"clinical_{col}"] = val

                records.append(row)
            except Exception as e:
                logger.warning("Failed to extract radiomics for %s: %s", case_id, e)

    # Process raw/paired DCE files
    elif len(acq0_files) > 0:
        if max_cases:
            acq0_files = acq0_files[:max_cases]

        for pre_p in tqdm(acq0_files, desc="Extracting Radiomics from DCE"):
            fname = pre_p.name
            pid = fname.split("_spy1")[0] if "_spy1" in fname else fname.split("_acq0")[0].split("_pre")[0]
            
            # Find matching T1 and Mask
            t1_name = fname.replace("_acq0", "_acq1").replace("_pre", "_post")
            t1_p = pre_p.parent / t1_name
            mask_p = (pre_p.parent.parent / "spy1_mask" / fname.replace("_acq0", "_mask")).with_suffix(".nii.gz")
            if not mask_p.exists():
                mask_cands = list(pre_p.parent.parent.glob(f"**/{pid}*mask.nii*"))
                if mask_cands:
                    mask_p = mask_cands[0]
                else:
                    continue

            if not t1_p.exists() or not mask_p.exists():
                continue

            try:
                t0_img = sitk.ReadImage(str(pre_p), sitk.sitkFloat32)
                t1_img = sitk.ReadImage(str(t1_p), sitk.sitkFloat32)
                mask_img = sitk.ReadImage(str(mask_p), sitk.sitkUInt8)

                # Resample and compute subtraction
                t0_iso = resample_to_isotropic(t0_img, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
                t1_iso = resample_to_isotropic(t1_img, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
                mask_iso = resample_to_isotropic(mask_img, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkNearestNeighbor)

                sub_iso = compute_subtraction_volume(t0_iso, t1_iso, clip_negative=True)
                feats = extractor.extract_features(sub_iso, mask_iso)

                row = {"pid": pid, **feats}
                if metadata_df is not None and pid in metadata_df.index:
                    meta_row = metadata_df.loc[pid].to_dict()
                    for col, val in meta_row.items():
                        if col != "pid":
                            row[f"clinical_{col}"] = val

                records.append(row)
            except Exception as e:
                logger.warning("Failed processing DCE case %s: %s", pid, e)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(out_p, index=False)
        logger.info("Successfully extracted radiomics for %d cases -> Saved to %s", len(df), out_p)
    else:
        logger.warning("No radiomics records were generated.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Radiomics Feature Extractor")
    parser.add_argument("--data_dir", type=str, default="data/segmentation_dryrun", help="Data directory")
    parser.add_argument("--output_csv", type=str, default="data/processed/radiomics_features.csv", help="Output CSV")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Metadata CSV path")
    args = parser.parse_args()

    extract_dataset_radiomics(
        data_dir=args.data_dir,
        output_csv_path=args.output_csv,
        metadata_csv=args.metadata_csv,
    )
