# AGENTS.md - Coding Standards & Architectural Guidelines

This document establishes the design principles, coding standards, and architectural conventions for the **3D DCE-MRI Breast Cancer Deep Learning Pipeline**. All human contributors and AI coding agents must adhere strictly to these rules.

---

## 1. Core Mission & Technology Stack

- **Domain**: 3D & 4D Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) for Breast Cancer Analysis (Deep Learning Volumetric Segmentation and Treatment Response Prediction).
- **Core Frameworks**:
  - **PyTorch** (`torch`): Deep learning foundation, volumetric modules, loss functions, backpropagation, and optimization loops.
  - **MONAI** (`monai`): Medical Open Network for AI for domain-specific medical data loaders, transforms, network architectures (3D UNet, DenseNet3D, ResNet3D), and metrics.
  - **SimpleITK & NiBabel**: NIfTI IO, physical space transformations, spacing/origin/direction handling, and resampling.
- **Pure Deep Learning Rule**: All predictive modeling must be performed exclusively with Deep Learning architectures (PyTorch / MONAI). No classical machine learning models (such as XGBoost, Random Forest, or SVM) are permitted.

---

## 2. Code Quality & Standards

### 2.1 Explicit Type Hinting
- Every function, method, and class attribute **must** have explicit type annotations.
- Utilize Python standard `typing` (`Optional`, `Union`, `List`, `Dict`, `Tuple`, `Callable`, `Sequence`, `Any`, `TypeVar`).
- For tensor and array representations, use explicit types:
  - `torch.Tensor` or `monai.data.MetaTensor` for PyTorch/MONAI tensors.
  - `np.ndarray` or `numpy.typing.NDArray[np.float32]` for NumPy arrays.
  - `sitk.Image` for SimpleITK images and `nib.Nifti1Image` for NiBabel objects.

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import monai
from monai.data import MetaTensor

def compute_subtraction_volume(
    pre_contrast: torch.Tensor,
    post_contrast: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compute peak subtraction volume from pre- and post-contrast DCE-MRI phases."""
    ...
```

### 2.2 Documentation & Docstrings
- Use **Google-style docstrings** for all modules, classes, public methods, and standalone functions.
- Include:
  - Concise summary of purpose.
  - `Args`: parameter names, types, and semantic descriptions (including expected tensor shapes, e.g., `(B, C, H, W, D)`).
  - `Returns`: return type, structure, and dimensionality.
  - `Raises`: explicit list of errors and exceptions that may be raised.

### 2.3 Exception Handling & Logging
- **No silent failures**: Avoid bare `except:` or empty `except Exception: pass`.
- **Use `logging`**: Always use standard Python `logging.getLogger(__name__)`. Do not use raw `print()` statements in library code (`src/`).
- **Contextual Logging**: Log input dimensions, physical voxel spacing, file paths, and memory states when errors occur.
- **Fail Fast & Informative**: Validate inputs early with clear assertion messages or domain-specific exceptions.

```python
import logging

logger = logging.getLogger(__name__)

try:
    image = sitk.ReadImage(str(file_path))
except Exception as e:
    logger.error("Failed to load NIfTI volume at %s: %s", file_path, e, exc_info=True)
    raise RuntimeError(f"Corrupted or missing NIfTI volume: {file_path}") from e
```

---

## 3. Medical Imaging & DCE-MRI Domain Rules

1. **Physical vs. Voxel Coordinates**:
   - Always preserve physical metadata (origin, spacing, direction matrix) when processing NIfTI volumes.
   - Resample images to a fixed voxel spacing (e.g., $1.0 \times 1.0 \times 1.0\text{ mm}^3$) using appropriate interpolators (`sitkLinear` / `Bilinear` for images, `sitkNearestNeighbor` for binary masks).
2. **Channel & Dimension Conventions**:
   - MONAI / PyTorch 3D standard format: `(Batch, Channel, Depth, Height, Width)` or `(B, C, H, W, D)` - keep consistent across transforms.
   - For 4D DCE-MRI time-series, distinguish temporal phases explicitly: `(B, TimePoints, H, W, D)` or stack timepoints along the channel dimension for multi-channel 3D networks.
3. **Data Partitioning & Leakage Prevention**:
   - Patient-level splitting is strictly required (`GroupKFold` or `StratifiedGroupKFold` by `PatientID`).
   - Slices or timepoints from the same subject must never appear across both train and validation/test sets.
4. **Reproducibility**:
   - Always set deterministic seeds (`monai.utils.set_determinism(seed=42)` and `torch.manual_seed(42)`).

---

## 4. Project Structure & Responsibilities

```text
├── data/
│   ├── raw/           # Original untouched NIfTI volumes and annotations
│   ├── processed/     # Resampled, cropped, and normalized arrays/cache
│   └── masks/         # Tumor segmentation masks and VOIs
├── src/
│   ├── preprocessing/ # Resampling, intensity normalization, 4D subtraction, kinetic curves
│   ├── models/        # PyTorch/MONAI architectures, loss functions, attention modules
│   └── utils/         # Metrics (Dice, AUC-ROC), logging, NIfTI IO, visualizers
├── notebooks/         # Exploratory analysis, radiomics extraction, experiments
├── AGENTS.md          # Coding standards (this document)
├── requirements.txt   # Pinned dependencies
└── README.md          # Project overview and instructions
```

- **`src/preprocessing/`**: Deterministic transforms, MONAI dictionary transforms, SimpleITK pipelines.
- **`src/models/`**: Modular network architectures inheriting from `torch.nn.Module`.
- **`src/utils/`**: Reusable helpers for file management, validation metrics, and tensor manipulation.
- **`notebooks/`**: Clean, modular experiments importing directly from `src.*`.

---

## 5. Testing & Verification

- Every new module should have corresponding unit tests in `tests/` or validation scripts in `src/utils/`.
- Ensure memory efficiency for 3D volumetric operations (use `torch.cuda.amp.autocast()` or `torch.amp.autocast('cuda')` when GPU is present and batch generator caching via `monai.data.CacheDataset`).
