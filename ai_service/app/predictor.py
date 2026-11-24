"""3D pCR response predictor with an illustrative Grad-CAM attribution map."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_service.app.config import (
    CLASSIFIER_CHECKPOINT,
    DEVICE,
    INPUT_SPATIAL_SIZE,
    MODEL_VERSION,
    PCR_THRESHOLD,
)
from ai_service.app.preprocessing import (
    normalize_volume,
    resize_volume_to_tensor,
    slice_to_base64_png,
)
from ai_service.app.schemas import PredictionResponse, SliceImage
from src.models.classifier3d import SimpleCNN3D

logger = logging.getLogger(__name__)


class DCEGradCAM:
    """3D Grad-CAM implementation for volumetric Convolutional Neural Networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Initialize Grad-CAM.

        Args:
            model: PyTorch 3D CNN model.
            target_layer: The target convolutional layer to extract activations and gradients from.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """Generate 3D Grad-CAM heatmap.

        Args:
            input_tensor: Tensor of shape (1, 1, D, H, W) on self.model device.
            class_idx: Target class index.

        Returns:
            np.ndarray: 3D Grad-CAM heatmap array of shape (D, H, W) normalized to [0, 1].
        """
        self.model.zero_grad()
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        target = logits[0, class_idx]
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured in Grad-CAM hooks.")

        # Global average pooling of gradients: (1, C, 1, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to focus on features that have a positive influence
        cam = F.relu(cam)

        # Upsample 3D CAM to the input spatial dimensions (D, H, W)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class BreastCancerPredictor:
    """Production predictor wrapping 3D DCE-MRI Deep Learning model and Explainable AI."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.device = DEVICE
        self.checkpoint_path = checkpoint_path or str(CLASSIFIER_CHECKPOINT)
        self.model: Optional[SimpleCNN3D] = None
        self.grad_cam: Optional[DCEGradCAM] = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading 3D DCE-MRI Classifier on device %s...", self.device)
        self.model = SimpleCNN3D(in_channels=1, num_classes=1, dropout=0.2)

        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(f"Trained classifier checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        logger.info("Successfully loaded weights from %s", self.checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        # The last conv layer in SimpleCNN3D is features[12]
        target_conv_layer = self.model.features[12]
        self.grad_cam = DCEGradCAM(self.model, target_conv_layer)

    def predict(
        self,
        volume: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> PredictionResponse:
        """Run research pCR response inference and create an illustrative attribution map.

        Args:
            volume: 3D numpy array of shape (X, Y, Z) or (H, W, D).
            voxel_spacing: Physical spacing in mm (dx, dy, dz).

        Returns:
            PredictionResponse: Research pCR estimate and attribution visualization.
        """
        start_time = time.perf_counter()

        # 1. Intensity Normalization
        norm_vol = normalize_volume(volume)

        # 2. Resample to standard network shape
        input_tensor = resize_volume_to_tensor(norm_vol, target_shape=INPUT_SPATIAL_SIZE).to(self.device)

        # 3. Model Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            pcr_probability = torch.sigmoid(logits).item()

        is_pcr = pcr_probability >= PCR_THRESHOLD
        pred_label = "pCR likely" if is_pcr else "pCR unlikely"
        confidence = pcr_probability if is_pcr else (1.0 - pcr_probability)

        # 4. Generate 3D Grad-CAM
        cam_available = True
        try:
            cam_3d = self.grad_cam.generate(input_tensor.clone(), class_idx=0)
        except Exception as e:
            logger.error("Grad-CAM generation failed: %s", e)
            cam_3d = np.zeros(INPUT_SPATIAL_SIZE, dtype=np.float32)
            cam_available = False

        # 5. Resize the attribution map to the canonical input volume dimensions.
        orig_shape_dhw = norm_vol.shape
        cam_orig = ndi.zoom(cam_3d, zoom=tuple(s / c for s, c in zip(orig_shape_dhw, cam_3d.shape)), order=1)

        # Grad-CAM is an attribution map, not a validated tumor segmentation.
        # 6. Identify the slice with the strongest attribution.
        # The input volume is ordered (X, Y, Z); sum over X and Y to score each axial Z slice.
        z_slice_sums = np.sum(cam_orig, axis=(0, 1))
        peak_z = int(np.argmax(z_slice_sums)) if len(z_slice_sums) > 0 else norm_vol.shape[2] // 2

        # 7. Extract representative slices with an illustrative attribution overlay.
        total_z = norm_vol.shape[2]
        key_slice_indices = sorted(list(set([
            max(0, total_z // 4),
            max(0, peak_z - 3),
            peak_z,
            min(total_z - 1, peak_z + 3),
            min(total_z - 1, (3 * total_z) // 4),
        ])))

        key_slices: List[SliceImage] = []
        for s_idx in key_slice_indices:
            # slice along Z dimension from norm_vol (X, Y, Z)
            slice_data = norm_vol[:, :, s_idx]
            cam_data = cam_orig[:, :, s_idx]

            b64_base = slice_to_base64_png(slice_data, heatmap_2d=None)
            b64_cam = slice_to_base64_png(slice_data, heatmap_2d=cam_data, alpha=0.55, colormap="jet")

            key_slices.append(
                SliceImage(
                    slice_index=s_idx,
                    plane="axial",
                    image_base64=b64_base,
                    gradcam_base64=b64_cam if cam_available else None,
                    is_peak_attribution=cam_available and s_idx == peak_z,
                )
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

        # 8. Research interpretation; avoid diagnosis and treatment recommendations.
        findings = [
            f"Research model output for pathological complete response: {pred_label}.",
            f"Predicted pCR probability: {pcr_probability:.3f}; decision threshold: {PCR_THRESHOLD:.2f}.",
            (
                f"Strongest Grad-CAM attribution is shown near axial slice {peak_z} of {total_z}; this is not a lesion contour."
                if cam_available else "Grad-CAM attribution was unavailable for this inference."
            ),
            "Prediction has not been clinically validated and must not be used for diagnosis or treatment decisions.",
        ]
        recommendations = ["Use only for research evaluation. Confirm treatment response with the established pathology workflow."]

        return PredictionResponse(
            prediction=pred_label,
            is_pcr_likely=is_pcr,
            confidence=round(confidence, 4),
            confidence_percentage=round(confidence * 100.0, 1),
            risk_level="Unvalidated",
            pcr_probability=round(pcr_probability, 4),
            tumor_volume_cm3=None,
            peak_slice_index=peak_z,
            total_slices=total_z,
            dimensions=[int(d) for d in norm_vol.shape],
            voxel_spacing=[round(float(s), 2) for s in voxel_spacing],
            model_version=MODEL_VERSION,
            processing_time_ms=elapsed_ms,
            key_slices=key_slices,
            findings=findings,
            recommendations=recommendations,
        )
