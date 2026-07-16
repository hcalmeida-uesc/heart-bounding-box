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


def compute_bounding_extent(mask: NDArray[Any]) -> NDArray[np.int64]:
    """Calcula a extensão do bounding box mínimo que envolve os voxels não-nulos (etapa 6).

    Segue a mesma convenção de eixos ``(Y, X, Z)`` de :func:`compute_geometric_centroid`.

    Args:
        mask: Volume 3D binário (ou com voxels não-nulos indicando o coração),
            tipicamente ``lcc_mask`` retornado por ``load_and_preprocess_image``.

    Returns:
        Array ``[extent_y, extent_x, extent_z]`` (int) com o número de voxels
        ocupados pelo bounding box mínimo em cada eixo (``max - min + 1``), na
        mesma resolução do volume de entrada.

    Raises:
        ValueError: se ``mask`` não for 3D ou não tiver nenhum voxel não-nulo.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"mask deve ser 3D, recebido shape={mask.shape}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError(
            "mask não contém nenhum voxel não-nulo; não é possível calcular a extensão"
        )

    return coords.max(axis=0) - coords.min(axis=0) + 1


def compute_bounding_box_bounds(
    mask: NDArray[Any],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Calcula os limites mínimo e máximo (inclusivos) do bbox que envolve os voxels não-nulos.

    Complementa :func:`compute_bounding_extent` quando os limites absolutos são necessários (não
    só a extensão) — por exemplo, para calcular o deslocamento de uma máscara em relação a um
    ponto de referência externo (ex.: âncora da aorta), em vez do bbox mínimo da própria máscara.
    Segue a mesma convenção de eixos ``(Y, X, Z)`` das demais funções deste módulo.

    Args:
        mask: Volume 3D binário.

    Returns:
        Tupla ``(min_coords, max_coords)``, cada uma um array ``[y, x, z]`` (int) com as
        coordenadas inclusivas do menor bounding box que contém todos os voxels não-nulos.

    Raises:
        ValueError: se ``mask`` não for 3D ou não tiver nenhum voxel não-nulo.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"mask deve ser 3D, recebido shape={mask.shape}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError(
            "mask não contém nenhum voxel não-nulo; não é possível calcular os limites"
        )

    return coords.min(axis=0), coords.max(axis=0)


__all__ = [
    "compute_geometric_centroid",
    "compute_bounding_extent",
    "compute_bounding_box_bounds",
]
