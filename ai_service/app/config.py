"""Configuration settings for Breast Cancer DCE-MRI Deep Learning AI Service."""

from __future__ import annotations

from pathlib import Path
import os
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"

CLASSIFIER_CHECKPOINT = CHECKPOINT_DIR / "best_3d_classifier.pt"
UNET_CHECKPOINT = CHECKPOINT_DIR / "best_monai_3dunet.pt"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
DEFAULT_BACKBONE = "simple_cnn3d"
INPUT_SPATIAL_SIZE = (128, 128, 64)  # (X, Y, Z) after canonical orientation and center crop/pad
PCR_THRESHOLD = 0.50
MODEL_VERSION = "pcr-research-v0.1"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]
