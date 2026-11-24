"""Environment and Hardware Verification Script for 3D DCE-MRI Deep Learning Pipeline.

Verifies PyTorch, MONAI 3D UNet and 3D Deep Classifier instantiation, SimpleITK,
NiBabel, GPU/CUDA availability, and scientific utility libraries.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SetupVerification")


def run_verification() -> Dict[str, Any]:
    """Verify deep learning libraries, 3D neural network architectures, and IO packages."""
    results: Dict[str, Any] = {}
    logger.info("Starting 3D DCE-MRI Deep Learning Pipeline Environment Verification...")

    # 1. PyTorch & Hardware Acceleration
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        results["cuda_device_count"] = torch.cuda.device_count()
        results["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "PyTorch successfully loaded: version %s (Device: %s)",
            torch.__version__,
            results["device"],
        )
    except Exception as e:
        logger.error("PyTorch verification failed: %s", e, exc_info=True)
        results["torch_version"] = f"FAILED: {e}"
        return results

    device = torch.device(results["device"])

    # 2. MONAI Deep Learning Architectures
    try:
        import monai
        from monai.networks.nets import UNet
        from src.models.classifier3d import build_3d_classifier

        results["monai_version"] = monai.__version__
        logger.info("MONAI successfully loaded: version %s", monai.__version__)

        # Test dummy 3D UNet model forward pass (Segmentation)
        logger.info("Testing 3D UNet segmentation model instantiation and dummy forward pass...")
        unet_model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=1,
        ).to(device)

        dummy_3d_input = torch.randn(1, 1, 32, 32, 16, device=device)
        with torch.no_grad():
            unet_output = unet_model(dummy_3d_input)

        results["monai_3d_unet_test"] = "PASSED"
        results["unet_input_shape"] = list(dummy_3d_input.shape)
        results["unet_output_shape"] = list(unet_output.shape)
        logger.info(
            "3D UNet forward pass verified: Input %s -> Output %s",
            tuple(dummy_3d_input.shape),
            tuple(unet_output.shape),
        )

        # Test dummy 3D Deep Classifier forward pass (Classification)
        logger.info("Testing 3D Deep Classifier model instantiation and dummy forward pass...")
        cls_model = build_3d_classifier(
            spatial_dims=3,
            in_channels=1,
            num_classes=1,
            backbone="simple_cnn3d",
            device=device,
        )
        with torch.no_grad():
            cls_output = cls_model(dummy_3d_input)

        results["monai_3d_classifier_test"] = "PASSED"
        results["cls_output_shape"] = list(cls_output.shape)
        logger.info(
            "3D Deep Classifier forward pass verified: Input %s -> Output Logits %s",
            tuple(dummy_3d_input.shape),
            tuple(cls_output.shape),
        )
    except Exception as e:
        logger.error("Deep Learning model verification failed: %s", e, exc_info=True)
        results["dl_models"] = f"FAILED: {e}"

    # 3. SimpleITK
    try:
        import SimpleITK as sitk
        results["simpleitk_version"] = sitk.__version__
        sample_img = sitk.Image([32, 32, 16], sitk.sitkFloat32)
        sample_img.SetSpacing([0.7, 0.7, 1.5])
        sample_img.SetOrigin([0.0, 0.0, 0.0])
        results["sitk_test"] = "PASSED"
        logger.info("SimpleITK loaded: version %s (3D Image creation test passed)", sitk.__version__)
    except Exception as e:
        logger.error("SimpleITK verification failed: %s", e, exc_info=True)
        results["simpleitk_version"] = f"FAILED: {e}"

    # 4. NiBabel
    try:
        import nibabel as nib
        import numpy as np
        results["nibabel_version"] = nib.__version__
        data = np.zeros((32, 32, 16), dtype=np.float32)
        affine = np.eye(4)
        nii = nib.Nifti1Image(data, affine)
        results["nibabel_test"] = "PASSED"
        logger.info("NiBabel loaded: version %s (NIfTI object creation test passed)", nib.__version__)
    except Exception as e:
        logger.error("NiBabel verification failed: %s", e, exc_info=True)
        results["nibabel_version"] = f"FAILED: {e}"

    # 5. Scientific Computing & Visualization
    try:
        import matplotlib
        import tqdm
        import pandas
        import scipy
        import jupyter

        results["matplotlib_version"] = matplotlib.__version__
        results["tqdm_version"] = tqdm.__version__
        results["pandas_version"] = pandas.__version__
        results["scipy_version"] = scipy.__version__

        logger.info("Scientific computing and utility libraries loaded successfully:")
        logger.info("  - matplotlib: %s", matplotlib.__version__)
        logger.info("  - pandas: %s", pandas.__version__)
        logger.info("  - scipy: %s", scipy.__version__)
        logger.info("  - tqdm: %s", tqdm.__version__)
    except Exception as e:
        logger.error("Scientific computing verification failed: %s", e, exc_info=True)
        results["scientific_utilities"] = f"FAILED: {e}"

    logger.info("Core Deep Learning verification complete!")
    return results


if __name__ == "__main__":
    summary = run_verification()
    print("\n" + "=" * 70)
    print("3D DCE-MRI DEEP LEARNING PIPELINE ENVIRONMENT VERIFICATION")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print("=" * 70)
