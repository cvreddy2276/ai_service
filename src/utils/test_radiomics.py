"""Test and verification script for Phase 4: Radiomics Feature Extraction.

Validates single-case extraction with DCEmriRadiomicsExtractor, verifies extraction
of 50+ quantitative radiomics metrics across 5 feature classes, executes batch extraction,
and exports data/processed/radiomics_features.csv.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import SimpleITK as sitk

from src.features.batch_extract import extract_dataset_radiomics
from src.features.radiomics_extractor import DCEmriRadiomicsExtractor
from src.preprocessing.mri_processor import compute_subtraction_volume, resample_to_isotropic
from src.utils.dummy_generator import create_synthetic_dce_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TestRadiomics")


def run_radiomics_tests() -> None:
    """Execute complete Phase 4 radiomics extraction test and batch generation."""
    logger.info("================================================================")
    logger.info("Starting Phase 4: PyRadiomics Feature Extraction Verification")
    logger.info("================================================================")

    # 1. Create a synthetic test case (T0, T1, Mask)
    logger.info("Creating synthetic 3D DCE-MRI case with focal tumor lesion...")
    t0_img, t1_img, mask_img = create_synthetic_dce_case(
        size=(192, 192, 48),
        spacing=(1.0, 1.0, 1.0),
        tumor_radius_voxels=(12, 12, 6),
        tumor_center_offset=(15, -10, 0),
        seed=42,
    )

    # 2. Compute isotropic subtraction volume
    sub_img = compute_subtraction_volume(t0_img, t1_img, clip_negative=True)
    logger.info("Subtraction Volume: Size=%s, Spacing=%s mm", sub_img.GetSize(), sub_img.GetSpacing())

    # 3. Instantiate DCEmriRadiomicsExtractor
    extractor = DCEmriRadiomicsExtractor(
        bin_width=25.0,
        enable_shape=True,
        enable_firstorder=True,
        enable_glcm=True,
        enable_glszm=True,
        enable_ngtdm=True,
    )

    # 4. Extract features
    logger.info("\nExtracting radiomics features from synthetic subtraction volume and lesion mask...")
    features = extractor.extract_features(sub_img, mask_img)

    num_features = len(features)
    logger.info("Total Radiomics Features Extracted: %d", num_features)
    assert num_features >= 50, f"Expected at least 50 features, got {num_features}"
    logger.info("Feature count requirement (>= 50): PASSED (%d features)", num_features)

    # 5. Group and categorize extracted features
    classes: Dict[str, Dict[str, float]] = {
        "Shape": {},
        "First-Order": {},
        "GLCM": {},
        "GLSZM": {},
        "NGTDM": {},
    }

    for k, v in sorted(features.items()):
        if "shape_" in k:
            classes["Shape"][k.replace("original_shape_", "")] = v
        elif "firstorder_" in k:
            classes["First-Order"][k.replace("original_firstorder_", "")] = v
        elif "glcm_" in k:
            classes["GLCM"][k.replace("original_glcm_", "")] = v
        elif "glszm_" in k:
            classes["GLSZM"][k.replace("original_glszm_", "")] = v
        elif "ngtdm_" in k:
            classes["NGTDM"][k.replace("original_ngtdm_", "")] = v

    logger.info("\n--- RADIOMICS FEATURE BREAKDOWN BY CLASS ---")
    for cls_name, cls_dict in classes.items():
        logger.info("  [%s: %d features]", cls_name, len(cls_dict))
        for feat_name, val in list(cls_dict.items())[:4]:  # Show first 4
            logger.info("    - %-32s: %.4f", feat_name, val)
        if len(cls_dict) > 4:
            logger.info("    ... (%d more %s features)", len(cls_dict) - 4, cls_name)

    # 6. Run Batch Extraction on Preprocessed Dataset
    data_dir = Path("data/segmentation_dryrun")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path("data/processed/radiomics_features.csv")
    logger.info("\nRunning Batch Extraction across dataset %s -> %s...", data_dir, output_csv)

    df = extract_dataset_radiomics(
        data_dir=data_dir,
        output_csv_path=output_csv,
        bin_width=25.0,
    )

    logger.info("\n--- BATCH EXTRACTION SUMMARY ---")
    logger.info("  Exported Table Shape: %s (Rows: %d, Columns: %d)", df.shape, df.shape[0], df.shape[1])
    logger.info("  Output CSV File: %s (Size: %.2f KB)", output_csv, output_csv.stat().st_size / 1024.0)

    # Assert non-empty dataframe
    assert len(df) > 0, "Extracted dataframe is empty"
    assert "pid" in df.columns, "Patient ID column missing from dataframe"
    logger.info("Batch dataframe verification: PASSED")

    logger.info("\n================================================================")
    logger.info("Phase 4: PyRadiomics Feature Extraction Verification PASSED!")
    logger.info("================================================================")


if __name__ == "__main__":
    run_radiomics_tests()
