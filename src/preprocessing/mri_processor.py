"""SimpleITK 3D DCE-MRI Preprocessing Module.

Provides physical-space medical image processing routines:
- Optimized N4 bias field correction with multi-resolution shrink factor
- Subtraction volume computation with spatial metadata alignment and clamping
- Isotropic spatial resampling preserving physical coordinates and bounding boxes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def apply_n4_bias_correction(
    image: sitk.Image,
    mask: Optional[sitk.Image] = None,
    shrink_factor: int = 2,
    num_iterations: Sequence[int] = (50, 50, 30, 20),
    convergence_threshold: float = 0.001,
) -> sitk.Image:
    """Correct MRI low-frequency magnetic field inhomogeneities using N4.

    Optimized using a spatial shrink factor during bias estimation before applying
    the reconstructed full-resolution log bias field back to the original image.

    Args:
        image: 3D input SimpleITK MRI volume.
        mask: Optional binary SimpleITK image defining the region of interest.
            If None, an Otsu threshold mask is automatically computed.
        shrink_factor: Downsampling factor for fast bias field estimation (default: 2).
        num_iterations: Number of iterations per resolution level (default: (50, 50, 30, 20)).
        convergence_threshold: Convergence energy threshold for N4 (default: 0.001).

    Returns:
        sitk.Image: Bias-field-corrected 3D SimpleITK image with float32 pixel type.

    Raises:
        ValueError: If input image dimension is not 3.
    """
    if image.GetDimension() != 3:
        raise ValueError(f"Expected 3D SimpleITK Image, got dimension {image.GetDimension()}")

    # 1. Ensure float32 casting
    input_image = sitk.Cast(image, sitk.sitkFloat32)

    # 2. Obtain or compute binary tissue mask
    if mask is None:
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
    else:
        mask_image = sitk.Cast(mask > 0, sitk.sitkUInt8)
        mask_image.CopyInformation(input_image)

    logger.debug(
        "Running N4 Bias Correction on volume of size %s with shrink_factor=%d",
        input_image.GetSize(),
        shrink_factor,
    )

    try:
        # 3. Downsample for fast bias field estimation if shrink_factor > 1
        if shrink_factor > 1:
            shrink_factors = [shrink_factor] * 3
            shrunk_image = sitk.Shrink(input_image, shrink_factors)
            shrunk_mask = sitk.Shrink(mask_image, shrink_factors)
        else:
            shrunk_image = input_image
            shrunk_mask = mask_image

        # 4. Execute N4 on shrunk image
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(list(num_iterations))
        corrector.SetConvergenceThreshold(convergence_threshold)
        _ = corrector.Execute(shrunk_image, shrunk_mask)

        # 5. Extract log bias field
        log_bias_shrunk = corrector.GetLogBiasFieldAsImage(shrunk_image)

        # 6. Resample log bias field back to full resolution
        if shrink_factor > 1:
            log_bias_full = sitk.Resample(
                log_bias_shrunk,
                input_image,
                sitk.Transform(),
                sitk.sitkBSpline,
                0.0,
                sitk.sitkFloat32,
            )
        else:
            log_bias_full = log_bias_shrunk

        # 7. Apply corrected log bias field: I_corrected = I / exp(log_bias)
        bias_field_full = sitk.Exp(log_bias_full)
        corrected_image = input_image / bias_field_full
        corrected_image.CopyInformation(input_image)

        return corrected_image

    except Exception as e:
        logger.warning(
            "N4 Bias Correction encountered an error: %s. Returning cast original image.",
            e,
            exc_info=True,
        )
        return input_image


def compute_subtraction_volume(
    pre_contrast: sitk.Image,
    post_contrast: sitk.Image,
    clip_negative: bool = True,
) -> sitk.Image:
    """Compute peak subtraction volume: I_sub = I_post - I_pre.

    Guarantees physical metadata matching between phases to prevent runtime errors
    and optionally clamps negative subtraction values to zero.

    Args:
        pre_contrast: 3D pre-contrast (T0) SimpleITK image.
        post_contrast: 3D post-contrast (T1) SimpleITK image.
        clip_negative: Whether to clamp negative values at 0.0 (default: True).

    Returns:
        sitk.Image: 3D SimpleITK subtraction volume (sitkFloat32).

    Raises:
        ValueError: If image dimensions do not match.
    """
    if pre_contrast.GetDimension() != post_contrast.GetDimension():
        raise ValueError(
            f"Dimension mismatch: pre={pre_contrast.GetDimension()} vs post={post_contrast.GetDimension()}"
        )

    # 1. Explicitly cast to sitkFloat32
    pre_float = sitk.Cast(pre_contrast, sitk.sitkFloat32)
    post_float = sitk.Cast(post_contrast, sitk.sitkFloat32)

    # 2. Align spatial metadata
    post_float.CopyInformation(pre_float)

    # 3. Compute voxel-wise subtraction
    subtraction = post_float - pre_float

    # 4. Optional non-negative clamping
    if clip_negative:
        subtraction = sitk.Clamp(subtraction, sitk.sitkFloat32, lowerBound=0.0, upperBound=1e8)

    subtraction.CopyInformation(pre_float)
    return subtraction


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """Resample a 3D SimpleITK image to isotropic voxel spacing.

    Calculates new grid dimensions to strictly preserve physical bounds:
    Size_new = round(Size_old * Spacing_old / Spacing_new)

    Args:
        image: Input 3D SimpleITK image.
        target_spacing: Desired voxel spacing (dx, dy, dz) in mm (default: (1.0, 1.0, 1.0)).
        interpolator: SimpleITK interpolator enum (e.g. sitk.sitkLinear or sitk.sitkNearestNeighbor).
        default_pixel_value: Fill value for out-of-bound voxels (default: 0.0).

    Returns:
        sitk.Image: Resampled 3D SimpleITK image.

    Raises:
        ValueError: If input dimension is not 3 or spacing values are non-positive.
    """
    if image.GetDimension() != 3:
        raise ValueError(f"Expected 3D image, got dimension {image.GetDimension()}")
    if any(s <= 0 for s in target_spacing):
        raise ValueError(f"Target spacing must be positive, got {target_spacing}")

    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()

    # Compute new output size preserving physical extent in mm
    new_size = [
        int(np.round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    # Ensure non-zero dimensions
    new_size = [max(1, s) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(list(target_spacing))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_pixel_value))
    resampler.SetOutputPixelType(image.GetPixelID())

    resampled_image = resampler.Execute(image)
    return resampled_image


class MRIProcessor:
    """High-level SimpleITK processor for 3D multi-temporal DCE-MRI cases."""

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        apply_n4: bool = True,
        shrink_factor: int = 2,
        clip_negative: bool = True,
    ) -> None:
        """Initialize MRIProcessor configuration.

        Args:
            target_spacing: Target isotropic resolution in mm (default: (1.0, 1.0, 1.0)).
            apply_n4: Whether to execute N4 bias correction on DCE phases.
            shrink_factor: N4 shrink factor optimization.
            clip_negative: Whether to clamp negative subtraction values.
        """
        self.target_spacing = target_spacing
        self.apply_n4 = apply_n4
        self.shrink_factor = shrink_factor
        self.clip_negative = clip_negative

    def process_case(
        self,
        t0_image: sitk.Image,
        t1_image: sitk.Image,
        mask_image: Optional[sitk.Image] = None,
    ) -> Dict[str, sitk.Image]:
        """Process a single DCE case through the complete SimpleITK pipeline.

        Steps:
        1. Cast and N4 Bias Field Correction (optional)
        2. Isotropic resampling (Linear for T0/T1, NearestNeighbor for mask)
        3. Subtraction volume computation (I_sub = I_t1 - I_t0)

        Args:
            t0_image: 3D pre-contrast SimpleITK image.
            t1_image: 3D post-contrast SimpleITK image.
            mask_image: Optional 3D binary lesion mask.

        Returns:
            Dict[str, sitk.Image]: Dictionary containing 't0', 't1', 'subtraction',
                and optional 'mask'.
        """
        # Step 1: N4 Bias Field Correction
        if self.apply_n4:
            t0_proc = apply_n4_bias_correction(
                t0_image,
                mask=mask_image,
                shrink_factor=self.shrink_factor,
            )
            t1_proc = apply_n4_bias_correction(
                t1_image,
                mask=mask_image,
                shrink_factor=self.shrink_factor,
            )
        else:
            t0_proc = sitk.Cast(t0_image, sitk.sitkFloat32)
            t1_proc = sitk.Cast(t1_image, sitk.sitkFloat32)

        # Step 2: Resample to isotropic spacing
        t0_resampled = resample_to_isotropic(
            t0_proc,
            target_spacing=self.target_spacing,
            interpolator=sitk.sitkLinear,
        )
        t1_resampled = resample_to_isotropic(
            t1_proc,
            target_spacing=self.target_spacing,
            interpolator=sitk.sitkLinear,
        )

        # Step 3: Compute Subtraction Volume
        subtraction = compute_subtraction_volume(
            pre_contrast=t0_resampled,
            post_contrast=t1_resampled,
            clip_negative=self.clip_negative,
        )

        results: Dict[str, sitk.Image] = {
            "t0": t0_resampled,
            "t1": t1_resampled,
            "subtraction": subtraction,
        }

        # Step 4: Resample Mask if present
        if mask_image is not None:
            mask_resampled = resample_to_isotropic(
                mask_image,
                target_spacing=self.target_spacing,
                interpolator=sitk.sitkNearestNeighbor,
                default_pixel_value=0.0,
            )
            results["mask"] = mask_resampled

        return results

    def process_files(
        self,
        t0_path: Union[str, Path],
        t1_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, sitk.Image]:
        """Load NIfTI files from disk, process, and optionally save results.

        Args:
            t0_path: Path to pre-contrast NIfTI file.
            t1_path: Path to post-contrast NIfTI file.
            mask_path: Optional path to tumor mask NIfTI file.
            output_dir: Optional directory to save resampled/subtracted NIfTI files.

        Returns:
            Dict[str, sitk.Image]: Dictionary of processed SimpleITK images.
        """
        logger.info("Loading T0 from %s and T1 from %s", t0_path, t1_path)
        t0_img = sitk.ReadImage(str(t0_path), sitk.sitkFloat32)
        t1_img = sitk.ReadImage(str(t1_path), sitk.sitkFloat32)
        mask_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8) if mask_path else None

        processed = self.process_case(t0_img, t1_img, mask_img)

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(processed["t0"], str(out_p / "t0_isotropic.nii.gz"))
            sitk.WriteImage(processed["t1"], str(out_p / "t1_isotropic.nii.gz"))
            sitk.WriteImage(processed["subtraction"], str(out_p / "subtraction_isotropic.nii.gz"))
            if "mask" in processed:
                sitk.WriteImage(processed["mask"], str(out_p / "mask_isotropic.nii.gz"))
            logger.info("Saved processed SimpleITK volumes to %s", out_p)

        return processed
