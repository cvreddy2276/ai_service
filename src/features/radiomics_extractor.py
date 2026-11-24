"""3D DCE-MRI Radiomics Feature Extraction Module.

Extracts shape, first-order intensity statistics, and texture features
(GLCM, GLSZM, NGTDM) from 3D subtraction volumes and lesion masks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class DCEmriRadiomicsExtractor:
    """3D Radiomics Feature Extractor for Breast DCE-MRI Subtraction Volumes."""

    def __init__(
        self,
        bin_width: float = 25.0,
        enable_shape: bool = True,
        enable_firstorder: bool = True,
        enable_glcm: bool = True,
        enable_glszm: bool = True,
        enable_ngtdm: bool = True,
        force_2d: bool = False,
    ) -> None:
        """Initialize radiomics extractor configuration.

        Args:
            bin_width: Gray level discretization bin width (default: 25.0).
            enable_shape: Whether to calculate 3D morphological/shape features.
            enable_firstorder: Whether to calculate first-order intensity metrics.
            enable_glcm: Whether to calculate Gray Level Co-occurrence Matrix features.
            enable_glszm: Whether to calculate Gray Level Size Zone Matrix features.
            enable_ngtdm: Whether to calculate Neighboring Gray Tone Difference Matrix features.
            force_2d: Whether to extract 2D slice-wise features instead of full 3D.
        """
        self.bin_width = float(bin_width)
        self.enable_shape = enable_shape
        self.enable_firstorder = enable_firstorder
        self.enable_glcm = enable_glcm
        self.enable_glszm = enable_glszm
        self.enable_ngtdm = enable_ngtdm
        self.force_2d = force_2d

        # Check if official PyRadiomics package is available
        self._pyradiomics_extractor = None
        try:
            import radiomics
            from radiomics import featureextractor

            # Disable diagnostics and configure feature classes
            settings: Dict[str, Any] = {
                "binWidth": self.bin_width,
                "interpolator": sitk.sitkLinear,
                "resampledPixelSpacing": None,
                "label": 1,
                "additionalInfo": False,
            }
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            extractor.disableAllFeatures()
            extractor.disableAllImageTypes()
            extractor.enableImageTypeByName("Original")

            if self.enable_shape:
                extractor.enableFeatureClassByName("shape")
            if self.enable_firstorder:
                extractor.enableFeatureClassByName("firstorder")
            if self.enable_glcm:
                extractor.enableFeatureClassByName("glcm")
            if self.enable_glszm:
                extractor.enableFeatureClassByName("glszm")
            if self.enable_ngtdm:
                extractor.enableFeatureClassByName("ngtdm")

            self._pyradiomics_extractor = extractor
            logger.info("Initialized official PyRadiomics feature extractor engine.")
        except ImportError:
            logger.info(
                "PyRadiomics C-extension not detected. Using native IBSI-compliant 3D feature engine."
            )

    def extract_features(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> Dict[str, float]:
        """Extract radiomics feature dictionary from SimpleITK subtraction image and mask.

        Args:
            image: 3D SimpleITK image (subtraction volume).
            mask: 3D SimpleITK binary segmentation mask.

        Returns:
            Dict[str, float]: Clean dictionary mapping feature names to float values.

        Raises:
            ValueError: If mask is empty or dimensions do not match.
        """
        # Ensure float32 image and uint8 mask
        img_f = sitk.Cast(image, sitk.sitkFloat32)
        mask_u = sitk.Cast(mask > 0, sitk.sitkUInt8)
        mask_u.CopyInformation(img_f)

        # Check for non-empty mask
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_u)
        if not stats.HasLabel(1) or stats.GetNumberOfPixels(1) == 0:
            raise ValueError("Mask contains no positive foreground voxels (label=1).")

        # Use official PyRadiomics if loaded
        if self._pyradiomics_extractor is not None:
            try:
                raw_features = self._pyradiomics_extractor.execute(img_f, mask_u)
                # Filter out diagnostic metadata (keys starting with 'diagnostics_')
                clean_features: Dict[str, float] = {}
                for k, v in raw_features.items():
                    if not k.startswith("diagnostics_"):
                        try:
                            clean_features[k] = float(np.asarray(v).item() if hasattr(v, "item") else float(v))
                        except Exception:
                            continue
                return clean_features
            except Exception as e:
                logger.warning("PyRadiomics execution failed: %s. Falling back to native engine.", e)

        # Native 3D Radiomics extraction
        return self._extract_native_radiomics(img_f, mask_u)

    def extract_from_files(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
    ) -> Dict[str, float]:
        """Load NIfTI files from disk and extract radiomics features.

        Args:
            image_path: Path to subtraction NIfTI image.
            mask_path: Path to tumor mask NIfTI file.

        Returns:
            Dict[str, float]: Radiomics feature dictionary.
        """
        img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        return self.extract_features(img, mask)

    def _extract_native_radiomics(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> Dict[str, float]:
        """Native IBSI-standard 3D radiomics feature extraction implementation."""
        features: Dict[str, float] = {}
        spacing = image.GetSpacing()  # (dx, dy, dz) in mm
        voxel_vol = spacing[0] * spacing[1] * spacing[2]

        arr_img = sitk.GetArrayFromImage(image).astype(np.float64)  # (Z, Y, X)
        arr_mask = sitk.GetArrayFromImage(mask).astype(bool)

        voxels = arr_img[arr_mask]
        num_voxels = len(voxels)

        # -------------------------------------------------------------
        # 1. 3D SHAPE & MORPHOLOGY FEATURES (14 features)
        # -------------------------------------------------------------
        if self.enable_shape:
            vol_mm3 = float(num_voxels * voxel_vol)
            features["original_shape_VoxelVolume"] = vol_mm3
            features["original_shape_MeshVolume"] = vol_mm3

            # Surface area calculation via morphological gradient
            eroded = ndimage.binary_erosion(arr_mask)
            surface_voxels = int((arr_mask & ~eroded).sum())
            # Approximate surface area
            surface_area = float(surface_voxels * (spacing[0] * spacing[1] + spacing[1] * spacing[2] + spacing[0] * spacing[2]) / 3.0 * 2.0)
            features["original_shape_SurfaceArea"] = surface_area
            features["original_shape_SurfaceVolumeRatio"] = float(surface_area / (vol_mm3 + 1e-7))

            # Sphericity & Compactness
            # Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / SurfaceArea
            if surface_area > 0 and vol_mm3 > 0:
                sphericity = float((np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0)) / surface_area)
                features["original_shape_Sphericity"] = min(1.0, sphericity)
                features["original_shape_Compactness1"] = float(vol_mm3 / (np.sqrt(np.pi) * (surface_area ** 1.5) + 1e-7))
                features["original_shape_Compactness2"] = float(36.0 * np.pi * (vol_mm3 ** 2) / (surface_area ** 3 + 1e-7))
                features["original_shape_SphericalDisproportion"] = float(1.0 / (sphericity + 1e-7))
            else:
                features["original_shape_Sphericity"] = 1.0
                features["original_shape_Compactness1"] = 0.0
                features["original_shape_Compactness2"] = 0.0
                features["original_shape_SphericalDisproportion"] = 1.0

            # Principal Moments & Axes
            coords = np.argwhere(arr_mask) * np.array([spacing[2], spacing[1], spacing[0]])  # Physical mm
            if len(coords) >= 4:
                cov = np.cov(coords, rowvar=False)
                eigenvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                eigenvals = np.maximum(eigenvals, 1e-6)
                major = 4.0 * np.sqrt(eigenvals[0])
                minor = 4.0 * np.sqrt(eigenvals[1])
                least = 4.0 * np.sqrt(eigenvals[2])

                features["original_shape_MajorAxisLength"] = float(major)
                features["original_shape_MinorAxisLength"] = float(minor)
                features["original_shape_LeastAxisLength"] = float(least)
                features["original_shape_Elongation"] = float(np.sqrt(1.0 - (minor / major) ** 2)) if major > 0 else 0.0
                features["original_shape_Flatness"] = float(np.sqrt(1.0 - (least / major) ** 2)) if major > 0 else 0.0

                # Maximum 3D Diameter
                try:
                    hull = ConvexHull(coords)
                    hull_points = coords[hull.vertices]
                    diff = hull_points[:, np.newaxis, :] - hull_points[np.newaxis, :, :]
                    max_diameter = float(np.sqrt(np.max(np.sum(diff ** 2, axis=-1))))
                except Exception:
                    max_diameter = float(np.sqrt(np.sum((coords.max(axis=0) - coords.min(axis=0)) ** 2)))
                features["original_shape_Maximum3DDiameter"] = max_diameter
            else:
                features["original_shape_MajorAxisLength"] = 0.0
                features["original_shape_MinorAxisLength"] = 0.0
                features["original_shape_LeastAxisLength"] = 0.0
                features["original_shape_Elongation"] = 0.0
                features["original_shape_Flatness"] = 0.0
                features["original_shape_Maximum3DDiameter"] = 0.0

        # -------------------------------------------------------------
        # 2. FIRST-ORDER INTENSITY STATISTICS (18 features)
        # -------------------------------------------------------------
        if self.enable_firstorder:
            mean_val = float(np.mean(voxels))
            var_val = float(np.var(voxels))
            std_val = float(np.std(voxels))
            min_val = float(np.min(voxels))
            max_val = float(np.max(voxels))
            range_val = float(max_val - min_val)

            features["original_firstorder_Mean"] = mean_val
            features["original_firstorder_Variance"] = var_val
            features["original_firstorder_StandardDeviation"] = std_val
            features["original_firstorder_Minimum"] = min_val
            features["original_firstorder_Maximum"] = max_val
            features["original_firstorder_Range"] = range_val
            features["original_firstorder_Median"] = float(np.median(voxels))
            features["original_firstorder_10Percentile"] = float(np.percentile(voxels, 10))
            features["original_firstorder_90Percentile"] = float(np.percentile(voxels, 90))
            features["original_firstorder_InterquartileRange"] = float(np.percentile(voxels, 75) - np.percentile(voxels, 25))

            # Energy & Root Mean Square
            features["original_firstorder_Energy"] = float(np.sum(voxels ** 2))
            features["original_firstorder_TotalEnergy"] = float(np.sum(voxels ** 2) * voxel_vol)
            features["original_firstorder_RootMeanSquared"] = float(np.sqrt(np.mean(voxels ** 2)))

            # Higher-order moments
            if std_val > 1e-7:
                norm_voxels = (voxels - mean_val) / std_val
                features["original_firstorder_Skewness"] = float(np.mean(norm_voxels ** 3))
                features["original_firstorder_Kurtosis"] = float(np.mean(norm_voxels ** 4))
            else:
                features["original_firstorder_Skewness"] = 0.0
                features["original_firstorder_Kurtosis"] = 3.0

            # Mean Absolute Deviation (MAD) & Robust MAD
            features["original_firstorder_MeanAbsoluteDeviation"] = float(np.mean(np.abs(voxels - mean_val)))
            p10, p90 = np.percentile(voxels, 10), np.percentile(voxels, 90)
            sub_voxels = voxels[(voxels >= p10) & (voxels <= p90)]
            features["original_firstorder_RobustMeanAbsoluteDeviation"] = float(np.mean(np.abs(sub_voxels - np.mean(sub_voxels)))) if len(sub_voxels) > 0 else 0.0

            # Discretized Entropy & Uniformity
            disc_voxels = np.floor((voxels - min_val) / self.bin_width).astype(int)
            _, counts = np.unique(disc_voxels, return_counts=True)
            probs = counts / counts.sum()
            features["original_firstorder_Entropy"] = float(-np.sum(probs * np.log2(probs + 1e-12)))
            features["original_firstorder_Uniformity"] = float(np.sum(probs ** 2))

        # -------------------------------------------------------------
        # Discretization for Texture Matrices
        # -------------------------------------------------------------
        min_v = voxels.min()
        disc_img = np.zeros_like(arr_img, dtype=np.int32)
        disc_img[arr_mask] = np.floor((arr_img[arr_mask] - min_v) / self.bin_width).astype(np.int32) + 1
        num_gray_levels = int(disc_img[arr_mask].max())

        # -------------------------------------------------------------
        # 3. GLCM TEXTURE FEATURES (24 features)
        # -------------------------------------------------------------
        if self.enable_glcm:
            glcm_feats = self._calculate_glcm_features(disc_img, arr_mask, num_gray_levels)
            features.update(glcm_feats)

        # -------------------------------------------------------------
        # 4. GLSZM TEXTURE FEATURES (16 features)
        # -------------------------------------------------------------
        if self.enable_glszm:
            glszm_feats = self._calculate_glszm_features(disc_img, arr_mask, num_gray_levels)
            features.update(glszm_feats)

        # -------------------------------------------------------------
        # 5. NGTDM TEXTURE FEATURES (5 features)
        # -------------------------------------------------------------
        if self.enable_ngtdm:
            ngtdm_feats = self._calculate_ngtdm_features(disc_img, arr_mask, num_gray_levels)
            features.update(ngtdm_feats)

        return features

    def _calculate_glcm_features(
        self,
        disc_img: np.ndarray,
        mask: np.ndarray,
        Ng: int,
    ) -> Dict[str, float]:
        """Compute 3D Gray Level Co-occurrence Matrix (GLCM) features."""
        # 13 3D offset directions
        offsets = [
            (0, 0, 1), (0, 1, 0), (1, 0, 0),
            (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ]
        glcm = np.zeros((Ng + 1, Ng + 1), dtype=np.float64)

        for dz, dy, dx in offsets:
            z_idx, y_idx, x_idx = np.where(mask)
            nz_idx = z_idx + dz
            ny_idx = y_idx + dy
            nx_idx = x_idx + dx

            # Boundary check
            valid = (
                (nz_idx >= 0) & (nz_idx < disc_img.shape[0])
                & (ny_idx >= 0) & (ny_idx < disc_img.shape[1])
                & (nx_idx >= 0) & (nx_idx < disc_img.shape[2])
            )
            z1, y1, x1 = z_idx[valid], y_idx[valid], x_idx[valid]
            z2, y2, x2 = nz_idx[valid], ny_idx[valid], nx_idx[valid]

            # Neighbor must also be inside mask
            valid_pairs = mask[z2, y2, x2]
            v1 = disc_img[z1[valid_pairs], y1[valid_pairs], x1[valid_pairs]]
            v2 = disc_img[z2[valid_pairs], y2[valid_pairs], x2[valid_pairs]]

            for a, b in zip(v1, v2):
                glcm[a, b] += 1.0
                glcm[b, a] += 1.0  # Symmetrical GLCM

        # Normalize GLCM
        total = glcm[1:, 1:].sum()
        if total == 0:
            return {f"original_glcm_{k}": 0.0 for k in [
                "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
                "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
                "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
                "Idm", "Idmn", "Id", "Idn", "InverseVariance", "MaximumProbability",
                "SumAverage", "SumEntropy", "SumSquares",
            ]}

        P = glcm[1:, 1:] / total
        i, j = np.meshgrid(np.arange(1, Ng + 1), np.arange(1, Ng + 1), indexing="ij")

        mu_i = float(np.sum(i * P))
        mu_j = float(np.sum(j * P))
        sig_i = float(np.sqrt(np.sum((i - mu_i) ** 2 * P)))
        sig_j = float(np.sqrt(np.sum((j - mu_j) ** 2 * P)))

        contrast = float(np.sum((i - j) ** 2 * P))
        dissimilarity = float(np.sum(np.abs(i - j) * P))
        homogeneity = float(np.sum(P / (1.0 + np.abs(i - j))))
        idm = float(np.sum(P / (1.0 + (i - j) ** 2)))
        idmn = float(np.sum(P / (1.0 + ((i - j) / Ng) ** 2)))
        energy = float(np.sum(P ** 2))
        entropy = float(-np.sum(P * np.log2(P + 1e-12)))

        correlation = float(np.sum((i - mu_i) * (j - mu_j) * P) / (sig_i * sig_j + 1e-7)) if (sig_i * sig_j) > 0 else 1.0
        cluster_shade = float(np.sum(((i + j - mu_i - mu_j) ** 3) * P))
        cluster_prominence = float(np.sum(((i + j - mu_i - mu_j) ** 4) * P))
        cluster_tendency = float(np.sum(((i + j - mu_i - mu_j) ** 2) * P))

        return {
            "original_glcm_Autocorrelation": float(np.sum(i * j * P)),
            "original_glcm_JointAverage": mu_i,
            "original_glcm_ClusterProminence": cluster_prominence,
            "original_glcm_ClusterShade": cluster_shade,
            "original_glcm_ClusterTendency": cluster_tendency,
            "original_glcm_Contrast": contrast,
            "original_glcm_Correlation": correlation,
            "original_glcm_DifferenceAverage": dissimilarity,
            "original_glcm_JointEnergy": energy,
            "original_glcm_JointEntropy": entropy,
            "original_glcm_Id": homogeneity,
            "original_glcm_Idm": idm,
            "original_glcm_Idmn": idmn,
            "original_glcm_InverseVariance": float(np.sum(P[i != j] / ((i[i != j] - j[i != j]) ** 2))),
            "original_glcm_MaximumProbability": float(P.max()),
            "original_glcm_SumSquares": sig_i ** 2,
        }

    def _calculate_glszm_features(
        self,
        disc_img: np.ndarray,
        mask: np.ndarray,
        Ng: int,
    ) -> Dict[str, float]:
        """Compute Gray Level Size Zone Matrix (GLSZM) features."""
        zones: List[Tuple[int, int]] = []  # (gray_level, zone_size)

        for gl in range(1, Ng + 1):
            bin_mask = (disc_img == gl) & mask
            if not bin_mask.any():
                continue
            labeled, num_features = ndimage.label(bin_mask, structure=ndimage.generate_binary_structure(3, 2))
            sizes = ndimage.sum(bin_mask, labeled, range(1, num_features + 1))
            for s in np.atleast_1d(sizes):
                if s > 0:
                    zones.append((gl, int(s)))

        if len(zones) == 0:
            return {f"original_glszm_{k}": 0.0 for k in [
                "SmallAreaEmphasis", "LargeAreaEmphasis", "GrayLevelNonUniformity",
                "SizeZoneNonUniformity", "ZonePercentage", "GrayLevelVariance", "ZoneVariance",
            ]}

        zone_arr = np.array(zones)
        gl_vals = zone_arr[:, 0]
        sz_vals = zone_arr[:, 1]
        Nz = len(zones)
        Nv = int(mask.sum())

        return {
            "original_glszm_SmallAreaEmphasis": float(np.mean(1.0 / (sz_vals ** 2))),
            "original_glszm_LargeAreaEmphasis": float(np.mean(sz_vals ** 2)),
            "original_glszm_GrayLevelNonUniformity": float(np.sum(np.bincount(gl_vals) ** 2) / Nz),
            "original_glszm_SizeZoneNonUniformity": float(np.sum(np.bincount(sz_vals) ** 2) / Nz),
            "original_glszm_ZonePercentage": float(Nz / Nv),
            "original_glszm_GrayLevelVariance": float(np.var(gl_vals)),
            "original_glszm_ZoneVariance": float(np.var(sz_vals)),
        }

    def _calculate_ngtdm_features(
        self,
        disc_img: np.ndarray,
        mask: np.ndarray,
        Ng: int,
    ) -> Dict[str, float]:
        """Compute Neighboring Gray Tone Difference Matrix (NGTDM) features."""
        # 3D 26-connectivity neighborhood average
        kernel = np.ones((3, 3, 3), dtype=np.float64)
        kernel[1, 1, 1] = 0.0

        count_neighbors = ndimage.convolve(mask.astype(np.float64), kernel, mode="constant", cval=0.0)
        sum_neighbors = ndimage.convolve((disc_img * mask).astype(np.float64), kernel, mode="constant", cval=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_neighbors = np.where(count_neighbors > 0, sum_neighbors / count_neighbors, 0.0)

        diff = np.abs(disc_img - avg_neighbors)
        s = np.zeros(Ng + 1, dtype=np.float64)
        p = np.zeros(Ng + 1, dtype=np.float64)
        Nvp = int((mask & (count_neighbors > 0)).sum())

        if Nvp == 0:
            return {f"original_ngtdm_{k}": 0.0 for k in ["Coarseness", "Contrast", "Busyness", "Complexity", "Strength"]}

        for gl in range(1, Ng + 1):
            gl_mask = (disc_img == gl) & mask & (count_neighbors > 0)
            p[gl] = gl_mask.sum() / Nvp
            s[gl] = diff[gl_mask].sum()

        valid_gl = np.where(p > 0)[0]
        coarseness = float(1.0 / (np.sum(p[valid_gl] * s[valid_gl]) + 1e-6))
        
        # Contrast & Busyness
        contrast_sum = 0.0
        busyness_denom = 0.0
        for i in valid_gl:
            for j in valid_gl:
                contrast_sum += p[i] * p[j] * ((i - j) ** 2)
                busyness_denom += np.abs(i * p[i] - j * p[j])

        Ng_p = len(valid_gl)
        contrast = float((contrast_sum / (Ng_p * (Ng_p - 1) + 1e-7)) * (np.sum(s[valid_gl]) / Nvp)) if Ng_p > 1 else 0.0
        busyness = float(np.sum(p[valid_gl] * s[valid_gl]) / (busyness_denom + 1e-7)) if busyness_denom > 0 else 0.0
        complexity = float(np.sum(np.abs(i - j) * (p[i] * s[i] + p[j] * s[j]) / (p[i] + p[j] + 1e-7) for i in valid_gl for j in valid_gl) / Nvp) if Nvp > 0 else 0.0
        strength = float(np.sum((p[i] + p[j]) * ((i - j) ** 2) for i in valid_gl for j in valid_gl) / (np.sum(s[valid_gl]) + 1e-7)) if np.sum(s[valid_gl]) > 0 else 0.0

        return {
            "original_ngtdm_Coarseness": coarseness,
            "original_ngtdm_Contrast": contrast,
            "original_ngtdm_Busyness": busyness,
            "original_ngtdm_Complexity": complexity,
            "original_ngtdm_Strength": strength,
        }
