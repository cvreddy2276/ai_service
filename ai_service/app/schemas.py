"""Pydantic schemas for DCE-MRI AI microservice."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str = "healthy"
    device: str
    model_loaded: bool
    model_version: str = "v1.2.0-dce-mri"
    pytorch_version: str


class SliceImage(BaseModel):
    slice_index: int
    plane: str  # 'axial', 'sagittal', 'coronal'
    image_base64: str
    gradcam_base64: Optional[str] = None
    is_peak_attribution: bool = False


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Research pCR estimate: 'pCR likely' or 'pCR unlikely'")
    is_pcr_likely: bool
    confidence: float = Field(..., description="Uncalibrated selected-class model score between 0.0 and 1.0")
    confidence_percentage: float
    risk_level: str = Field(..., description="Unvalidated decision category; not a clinical risk score")
    pcr_probability: float
    tumor_volume_cm3: Optional[float] = Field(None, description="Not calculated; Grad-CAM is not segmentation")
    peak_slice_index: int
    total_slices: int
    dimensions: List[int]
    voxel_spacing: List[float]
    model_version: str = "pcr-research-v0.1"
    processing_time_ms: float
    key_slices: List[SliceImage]
    findings: List[str]
    recommendations: List[str]


class GradCamRequest(BaseModel):
    slice_index: int
    plane: str = "axial"
    alpha: float = 0.45
    colormap: str = "jet"


class ModelMetricsResponse(BaseModel):
    model_name: str
    backbone: str
    dataset: str
    accuracy: float
    roc_auc: float
    sensitivity_recall: float
    specificity: float
    precision: float
    f1_score: float
    balanced_accuracy: float
    dice_score_segmentation: float
    confusion_matrix: Dict[str, int]
    slice_thickness_mm: float
    target_voxel_spacing: List[float]


class ModelMetricsUnavailable(BaseModel):
    status: str = "unavailable"
    detail: str


class DemoScanInfo(BaseModel):
    id: str
    name: str
    description: str
    sample_type: str
    file_path: str
    size_mb: float
