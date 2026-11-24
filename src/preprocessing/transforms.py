"""Custom MONAI/SimpleITK dictionary transforms for 3D DCE-MRI preprocessing.

Includes subtraction volume calculation, N4 bias field correction,
Z-score intensity normalization, isotropic resampling, and tumor ROI cropping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)


class ComputeSubtractiond(MapTransform):
    """Compute peak subtraction volume (I_post - I_pre) from DCE-MRI phases.

    Optionally clips negative values and creates a multi-channel DCE tensor
    combining [pre, post, subtraction].
    """

    def __init__(
        self,
        pre_key: str = "t0",
        post_key: str = "t1",
        sub_key: str = "subtraction",
        multichannel_key: Optional[str] = "image",
        stack_phases: bool = True,
        clip_negative: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize subtraction transform.

        Args:
            pre_key: Dictionary key for pre-contrast volume (T0).
            post_key: Dictionary key for post-contrast volume (T1).
            sub_key: Destination dictionary key for computed subtraction volume.
            multichannel_key: Destination key for the model image tensor.
            stack_phases: Whether to stack [T0, T1, subtraction] or use subtraction alone.
            clip_negative: Whether to clip negative subtraction values to zero.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys=[pre_key, post_key], allow_missing_keys=allow_missing_keys)
        self.pre_key = pre_key
        self.post_key = post_key
        self.sub_key = sub_key
        self.multichannel_key = multichannel_key
        self.stack_phases = stack_phases
        self.clip_negative = clip_negative

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply subtraction to dictionary data.

        Args:
            data: Input dictionary containing pre and post contrast tensors.

        Returns:
            Dict[str, Any]: Dictionary updated with subtraction volume.
        """
        d = dict(data)
        if self.pre_key not in d or self.post_key not in d:
            if not self.allow_missing_keys:
                raise KeyError(f"Keys '{self.pre_key}' and '{self.post_key}' are required for subtraction.")
            return d

        pre = d[self.pre_key]
        post = d[self.post_key]

        if tuple(pre.shape) != tuple(post.shape):
            raise ValueError(
                f"Pre- and post-contrast shapes differ ({tuple(pre.shape)} vs {tuple(post.shape)}); "
                "resample and align phases before subtraction."
            )
        pre_affine = getattr(pre, "affine", None)
        post_affine = getattr(post, "affine", None)
        if pre_affine is not None and post_affine is not None:
            if not torch.allclose(torch.as_tensor(pre_affine), torch.as_tensor(post_affine), atol=1e-3, rtol=0):
                raise ValueError("Pre- and post-contrast physical affines differ; register or align phases before subtraction.")

        # Compute subtraction only after validating voxel and physical alignment.
        sub = post - pre
        if self.clip_negative:
            if isinstance(sub, torch.Tensor):
                sub = torch.clamp(sub, min=0.0)
            else:
                sub = np.clip(sub, 0.0, None)

        d[self.sub_key] = sub

        if self.multichannel_key is not None:
            if not self.stack_phases:
                d[self.multichannel_key] = sub
                return d
            # Stack [pre, post, sub] along channel dimension (dim 0)
            if isinstance(pre, torch.Tensor):
                # Ensure each has channel dim
                c_pre = pre if pre.ndim == 4 else pre.unsqueeze(0)
                c_post = post if post.ndim == 4 else post.unsqueeze(0)
                c_sub = sub if sub.ndim == 4 else sub.unsqueeze(0)
                d[self.multichannel_key] = torch.cat([c_pre, c_post, c_sub], dim=0)
            else:
                c_pre = pre if pre.ndim == 4 else np.expand_dims(pre, axis=0)
                c_post = post if post.ndim == 4 else np.expand_dims(post, axis=0)
                c_sub = sub if sub.ndim == 4 else np.expand_dims(sub, axis=0)
                d[self.multichannel_key] = np.concatenate([c_pre, c_post, c_sub], axis=0)

        return d


class N4BiasFieldCorrectiond(MapTransform):
    """Apply SimpleITK N4 Bias Field Correction to MRI volumes."""

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: Optional[str] = None,
        num_iterations: Sequence[int] = (50, 50, 30, 20),
        convergence_threshold: float = 0.001,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize N4 Bias Field Correction transform.

        Args:
            keys: Dictionary keys of volumes to correct.
            mask_key: Optional key for brain/breast mask to restrict correction.
            num_iterations: Number of iterations per resolution level.
            convergence_threshold: Convergence energy threshold.
            allow_missing_keys: Whether to ignore missing dictionary keys.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.mask_key = mask_key
        self.num_iterations = list(num_iterations)
        self.convergence_threshold = convergence_threshold

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply N4 correction to specified dictionary keys.

        Args:
            data: Input dictionary with tensor or array images.

        Returns:
            Dict[str, Any]: Dictionary with bias-corrected volumes.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            # Extract numpy array
            is_torch = isinstance(val, torch.Tensor)
            meta = val.meta if isinstance(val, MetaTensor) else None
            arr = val.cpu().numpy() if is_torch else np.array(val)
            
            # Squeeze channel dim if present
            has_channel = arr.ndim == 4 and arr.shape[0] == 1
            if has_channel:
                arr = arr[0]

            try:
                sitk_img = sitk.GetImageFromArray(arr.astype(np.float32))
                
                # Create foreground mask using Otsu threshold if no mask provided
                if self.mask_key and self.mask_key in d:
                    mask_arr = d[self.mask_key]
                    if isinstance(mask_arr, torch.Tensor):
                        mask_arr = mask_arr.cpu().numpy()
                    if mask_arr.ndim == 4 and mask_arr.shape[0] == 1:
                        mask_arr = mask_arr[0]
                    sitk_mask = sitk.GetImageFromArray((mask_arr > 0).astype(np.uint8))
                else:
                    sitk_mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetMaximumNumberOfIterations(self.num_iterations)
                corrector.SetConvergenceThreshold(self.convergence_threshold)
                corrected_img = corrector.Execute(sitk_img, sitk_mask)
                corrected_arr = sitk.GetArrayFromImage(corrected_img)

                if has_channel:
                    corrected_arr = np.expand_dims(corrected_arr, axis=0)

                if is_torch:
                    if meta is not None:
                        d[key] = MetaTensor(corrected_arr, meta=meta)
                    else:
                        d[key] = torch.from_numpy(corrected_arr).to(val.device)
                else:
                    d[key] = corrected_arr
            except Exception as e:
                logger.warning(
                    "N4 Bias Correction failed for key '%s': %s. Falling back to original volume.",
                    key,
                    e,
                )
        return d


class ZScoreNormalizeIntensityd(MapTransform):
    """Z-score intensity normalization computed over non-zero foreground voxels."""

    def __init__(
        self,
        keys: KeysCollection,
        channel_wise: bool = True,
        nonzero: bool = True,
        eps: float = 1e-7,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize Z-score intensity normalizer.

        Args:
            keys: Dictionary keys to normalize.
            channel_wise: Whether to normalize each channel independently.
            nonzero: Whether to calculate mean and std only over non-zero voxels.
            eps: Numerical epsilon to avoid division by zero.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.channel_wise = channel_wise
        self.nonzero = nonzero
        self.eps = eps

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply non-zero Z-score normalization.

        Args:
            data: Input dictionary.

        Returns:
            Dict[str, Any]: Dictionary with normalized tensors.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            if isinstance(val, torch.Tensor):
                d[key] = self._normalize_tensor(val)
            else:
                d[key] = self._normalize_numpy(val)
        return d

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize PyTorch / MetaTensor."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        normalized = tensor.clone()
        num_channels = tensor.shape[0] if self.channel_wise else 1

        for c in range(num_channels):
            channel_data = tensor[c] if self.channel_wise else tensor
            mask = channel_data != 0 if self.nonzero else torch.ones_like(channel_data, dtype=torch.bool)
            
            if mask.sum() > 0:
                mean = channel_data[mask].mean()
                std = channel_data[mask].std()
                if std > self.eps:
                    if self.channel_wise:
                        normalized[c] = torch.where(mask, (channel_data - mean) / (std + self.eps), torch.zeros_like(channel_data))
                    else:
                        normalized = torch.where(mask, (tensor - mean) / (std + self.eps), torch.zeros_like(tensor))
                else:
                    if self.channel_wise:
                        normalized[c] = torch.where(mask, channel_data - mean, torch.zeros_like(channel_data))
                    else:
                        normalized = torch.where(mask, tensor - mean, torch.zeros_like(tensor))

        return normalized.squeeze(0) if squeeze_back else normalized

    def _normalize_numpy(self, arr: np.ndarray) -> np.ndarray:
        """Normalize NumPy array."""
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
            squeeze_back = True
        else:
            squeeze_back = False

        normalized = arr.copy().astype(np.float32)
        num_channels = arr.shape[0] if self.channel_wise else 1

        for c in range(num_channels):
            channel_data = arr[c] if self.channel_wise else arr
            mask = channel_data != 0 if self.nonzero else np.ones_like(channel_data, dtype=bool)
            
            if mask.sum() > 0:
                mean = channel_data[mask].mean()
                std = channel_data[mask].std()
                if std > self.eps:
                    norm_val = (channel_data - mean) / (std + self.eps)
                else:
                    norm_val = channel_data - mean
                
                if self.channel_wise:
                    normalized[c] = np.where(mask, norm_val, 0.0)
                else:
                    normalized = np.where(mask, norm_val, 0.0)

        return np.squeeze(normalized, axis=0) if squeeze_back else normalized


class TumorRoiCropd(MapTransform):
    """Crop a fixed spatial ROI (e.g. 128x128x64) around the tumor mask centroid.

    If mask is empty or absent, falls back to center-of-volume cropping.
    Handles padding if the target ROI extends beyond the image bounds.
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Sequence[int] = (128, 128, 64),
        mask_key: Optional[str] = "mask",
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize ROI crop transform.

        Args:
            keys: Dictionary keys to crop (e.g. ['image', 't0', 't1', 'mask']).
            roi_size: Desired output spatial dimensions (H, W, D).
            mask_key: Key of the mask used to locate tumor centroid.
            allow_missing_keys: Whether to ignore missing keys.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.roi_size = tuple(roi_size)
        self.mask_key = mask_key

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        """Crop specified spatial ROI around tumor centroid.

        Args:
            data: Input dictionary containing 3D/4D volumes and mask.

        Returns:
            Dict[str, Any]: Dictionary with cropped volumes of shape (C, *roi_size).
        """
        d = dict(data)
        first_key = list(self.key_iterator(d))[0]
        ref_shape = d[first_key].shape  # (C, H, W, D) or (H, W, D)
        spatial_shape = ref_shape[1:] if len(ref_shape) == 4 else ref_shape

        # Find centroid from mask if available
        center = self._find_center(d, spatial_shape)

        # Compute crop bounding box [start, end] for each spatial dimension
        slices: List[slice] = []
        pad_needed: List[Tuple[int, int]] = []

        for dim in range(3):
            size = self.roi_size[dim]
            c = center[dim]
            start = int(c - size // 2)
            end = start + size
            
            # Boundary checks and padding calculation
            pad_before = max(0, -start)
            pad_after = max(0, end - spatial_shape[dim])
            pad_needed.append((pad_before, pad_after))
            
            # Valid slice bounds inside original array
            valid_start = max(0, start)
            valid_end = min(spatial_shape[dim], end)
            slices.append(slice(valid_start, valid_end))

        # Crop and pad each key
        for key in self.key_iterator(d):
            val = d[key]
            is_torch = isinstance(val, torch.Tensor)
            meta = val.meta if isinstance(val, MetaTensor) else None
            
            if is_torch:
                has_channel = val.ndim == 4
                if not has_channel:
                    val = val.unsqueeze(0)
                
                # Crop valid portion
                cropped = val[:, slices[0], slices[1], slices[2]]
                
                # Pad if necessary: F.pad expects (pad_D_left, pad_D_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right)
                needs_padding = any(p[0] > 0 or p[1] > 0 for p in pad_needed)
                if needs_padding:
                    # PyTorch pad orders reverse spatial: dim 2, dim 1, dim 0
                    torch_pad = (
                        pad_needed[2][0], pad_needed[2][1],
                        pad_needed[1][0], pad_needed[1][1],
                        pad_needed[0][0], pad_needed[0][1],
                    )
                    cropped = torch.nn.functional.pad(cropped, torch_pad, mode="constant", value=0.0)

                if meta is not None:
                    d[key] = MetaTensor(cropped, meta=meta)
                else:
                    d[key] = cropped if has_channel else cropped.squeeze(0)
            else:
                has_channel = val.ndim == 4
                if not has_channel:
                    val = np.expand_dims(val, axis=0)
                
                cropped = val[:, slices[0], slices[1], slices[2]]
                needs_padding = any(p[0] > 0 or p[1] > 0 for p in pad_needed)
                if needs_padding:
                    np_pad = ((0, 0), pad_needed[0], pad_needed[1], pad_needed[2])
                    cropped = np.pad(cropped, np_pad, mode="constant", constant_values=0)
                
                d[key] = cropped if has_channel else np.squeeze(cropped, axis=0)

        return d

    def _find_center(self, data: Mapping[str, Any], spatial_shape: Sequence[int]) -> Tuple[int, int, int]:
        """Find center coordinates in (H, W, D)."""
        if self.mask_key and self.mask_key in data:
            mask = data[self.mask_key]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            if mask_np.ndim == 4:
                mask_np = mask_np[0]

            pos = np.argwhere(mask_np > 0)
            if len(pos) > 0:
                center = tuple(int(np.round(np.mean(pos[:, i]))) for i in range(3))
                return (center[0], center[1], center[2])

        # Default: center of volume
        return (spatial_shape[0] // 2, spatial_shape[1] // 2, spatial_shape[2] // 2)
