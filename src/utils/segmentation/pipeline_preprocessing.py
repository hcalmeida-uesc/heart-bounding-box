"""Carregamento e pré-processamento reutilizáveis no pipeline de bounding box.

Este módulo concentra a etapa que transforma um caso ImageCAS bruto em um
volume cardíaco 3D (downscale + threshold HU + LCC por fatia), base para o
cálculo do bounding box determinístico (Capítulo 2 da tese).
"""

from typing import Any, Dict

import cv2
import numpy as np

from ..processing.preprocessing import (
    downscale_image_ndi,
    downscale_image_opencv,
    largest_connected_component,
    threshold_image_with_offset,
)
from .lower_threshold import resolve_lower_threshold
from ..utils.nifti_io import load_raw_img_and_label


def load_and_preprocess_image(
    img_id: str, base_path: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Carrega imagem/label, reduz resolução e monta a maior componente conectada."""
    # Carrega o volume CCTA e a máscara de referência do ImageCAS.
    nii_img, nii_label = load_raw_img_and_label(
        f"{base_path}/{img_id}.img.nii.gz", f"{base_path}/{img_id}.label.nii.gz"
    )
    spacing = nii_img.header.get_zooms()
    img = np.array(nii_img.get_fdata(), dtype=np.float32)
    label = np.array(nii_label.get_fdata()).astype(np.uint8)

    # Define o método de downsampling usado para imagem e label.
    downscale_factors = config["DOWNSCALE_FACTORS"]
    use_opencv = config["DOWNSCALE_METHOD"] == "opencv"
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    opencv_interpolation = interpolation_map.get(
        config["OPENCV_INTERPOLATION"], cv2.INTER_LINEAR
    )

    if use_opencv:
        down_image = downscale_image_opencv(
            img,
            downscale_factors,
            interpolation=opencv_interpolation,
        )
    else:
        down_image = downscale_image_ndi(img, downscale_factors, order=3)

    min_threshold, lower_threshold_details = resolve_lower_threshold(
        down_image,
        config,
    )
    thresh_vals = (
        float(min_threshold),
        float(np.percentile(down_image, config["MAX_THRESHOLD_PERCENTILE"])),
    )
    thresh_image, thresh_mask, offset = threshold_image_with_offset(
        down_image,
        *thresh_vals,
    )

    if config.get("LCC_PER_SLICE", True):
        lcc_image = np.zeros_like(thresh_image, dtype=thresh_image.dtype)
        lcc_mask = np.zeros_like(thresh_mask, dtype=bool)
        for z_idx in range(thresh_image.shape[2]):
            lcc_slice, lcc_mask_slice = largest_connected_component(
                thresh_image[:, :, z_idx],
                thresh_mask[:, :, z_idx],
            )
            lcc_image[:, :, z_idx] = lcc_slice
            lcc_mask[:, :, z_idx] = lcc_mask_slice
    else:
        lcc_image, lcc_mask = largest_connected_component(thresh_image, thresh_mask)

    lcc_image = (lcc_image - offset).astype(np.float32)
    preprocessing_details = {
        **lower_threshold_details,
        "min_threshold": float(thresh_vals[0]),
        "max_threshold": float(thresh_vals[1]),
        "threshold_voxels": int(np.sum(thresh_mask)),
        "lcc_voxels": int(np.sum(lcc_mask)),
    }

    # O label usa interpolação de vizinho mais próximo para preservar classes.
    label = downscale_image_ndi(label, downscale_factors, order=0)

    dx, dy, dz = (
        spacing[0] * downscale_factors[0],
        spacing[1] * downscale_factors[1],
        spacing[2] * downscale_factors[2],
    )

    return {
        "lcc_image": lcc_image,
        "lcc_mask": lcc_mask,
        "label": label,
        "preprocessing_details": preprocessing_details,
        "spacing": spacing,
        "scaled_spacing": (dx, dy, dz),
        "downscale_factors": downscale_factors,
    }


__all__ = ["load_and_preprocess_image"]
