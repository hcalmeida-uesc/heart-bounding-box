"""Etapas do bounding box cardíaco determinístico (Capítulo 2), a partir da etapa 5.

Este módulo concentra a lógica nova (não herdada do pipeline de segmentação
arterial): centróide geométrico, envelope de dimensões, posicionamento, corte
e validação de cobertura/Rv. Consome a saída de
``pipeline_preprocessing.load_and_preprocess_image`` (etapas 1-4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_geometric_centroid(mask: NDArray[Any]) -> NDArray[np.float64]:
    """Calcula o centróide geométrico de uma máscara binária 3D (etapa 5).

    Média aritmética das coordenadas dos voxels não-nulos, por eixo, seguindo
    a convenção ``(Y, X, Z)`` usada em todo o pipeline (eixo 0 = altura/y,
    eixo 1 = largura/x, eixo 2 = fatia axial/z).

    Args:
        mask: Volume 3D binário (ou com voxels não-nulos indicando o coração),
            tipicamente ``lcc_mask`` retornado por
            ``load_and_preprocess_image``.

    Returns:
        Array ``[y_c, x_c, z_c]`` (float64) com o centróide em coordenadas de
        voxel, na mesma resolução do volume de entrada.

    Raises:
        ValueError: se ``mask`` não for 3D ou não tiver nenhum voxel não-nulo.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"mask deve ser 3D, recebido shape={mask.shape}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError(
            "mask não contém nenhum voxel não-nulo; não é possível calcular o centróide"
        )

    return coords.mean(axis=0)


__all__ = ["compute_geometric_centroid"]
