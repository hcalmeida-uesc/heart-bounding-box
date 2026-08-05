"""Etapas 7-8 do bounding box cardíaco: posicionamento (offset por percentil, resolução
reduzida) e corte em resolução integral (imagem + label originais, sem interpolação).

Etapas 1-6 (`aorta_anchor.py`) calculam âncora e offset em resolução reduzida a partir de
``all_cases.csv``. Este módulo converte esse resultado em índices de corte na imagem/label
originais (resolução integral) e realiza o corte propriamente dito, preservando o affine/origem
do NIfTI corretamente (fórmula geral, não assume affine diagonal positivo — ver
:func:`crop_volume_and_update_affine`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def compute_percentile_offsets(
    records_df: pd.DataFrame, offset_percentile: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Offset mínimo/máximo por eixo, no percentil ``offset_percentile`` da distribuição
    ``label_{axis}_min/max - anchor_{axis}`` de ``records_df`` (resolução reduzida).

    Mesma fórmula computada internamente por
    :func:`utils.segmentation.aorta_anchor.evaluate_percentile_bbox_tradeoff` (que não a expõe
    no dict de retorno, só as métricas agregadas) — extraída aqui para reaproveitar sem duplicar
    a lógica quando os offsets em si (não só `Rv`/perda agregados) são necessários, como no
    posicionamento e corte em resolução integral (etapas 7-8). ``offset_percentile=100``
    reproduz o mínimo/máximo absoluto de
    :func:`utils.segmentation.aorta_anchor.calibrate_bbox_offsets`.

    ``records_df`` deve já vir filtrado pelo chamador (ex.: só o split de treino, já filtrado por
    um critério de exclusão) — esta função não filtra nada, só agrega por percentil.
    """
    if not (50 < offset_percentile <= 100):
        raise ValueError(
            f"offset_percentile deve estar em (50, 100], recebido {offset_percentile}"
        )
    required_columns = {
        "anchor_y", "anchor_x", "anchor_z",
        "label_y_min", "label_y_max",
        "label_x_min", "label_x_max",
        "label_z_min", "label_z_max",
    }
    missing = required_columns - set(records_df.columns)
    if missing:
        raise ValueError(f"records_df não contém as colunas esperadas: {sorted(missing)}")
    if records_df.empty:
        raise ValueError("records_df está vazio -- não é possível calcular percentis.")

    offset_min: dict[str, float] = {}
    offset_max: dict[str, float] = {}
    for axis in ("y", "x", "z"):
        anchor_col = records_df[f"anchor_{axis}"]
        low_values = records_df[f"label_{axis}_min"] - anchor_col
        high_values = records_df[f"label_{axis}_max"] - anchor_col
        offset_min[axis] = float(np.percentile(low_values, 100.0 - offset_percentile))
        offset_max[axis] = float(np.percentile(high_values, offset_percentile))
    return offset_min, offset_max


def compute_full_resolution_bbox(
    anchor_reduced: tuple[float, float, float],
    offset_min_reduced: dict[str, float],
    offset_max_reduced: dict[str, float],
    downscale_factors: tuple[float, float, float],
    volume_shape_full: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Converte âncora+offset (resolução reduzida, convenção ``(y, x, z)``) em índices de corte
    na resolução integral.

    ``bbox_min/max_reduced = anchor + offset`` primeiro (mesma convenção de sinal usada em todo
    o módulo `aorta_anchor`), depois escalado por ``downscale_factors`` — mesma relação usada em
    ``offset_full = offset_reduzido * fator`` (`calibrate_bbox_offsets`,
    `evaluate_percentile_bbox_tradeoff`), aplicada aqui também à âncora (posição absoluta, não só
    a diferença) porque o downscale do pipeline é uma reamostragem por fator fixo sem deslocamento
    de origem (confirmado empiricamente em `docs/step6v2_aorta_anchor.md`, "Esclarecimento 1":
    reconstrução exata do shape original em 932/932 casos checados).

    Arredondamento **conservador**: ``floor`` no mínimo, ``ceil`` no máximo — nunca encolhe a
    cobertura pretendida por causa de arredondamento de ponto flutuante. Limites finais recortados
    (clamp) para ``[0, shape_full]`` por eixo. Levanta ``ValueError`` se o bbox ficar vazio após o
    clamp em algum eixo (âncora fora do volume, por exemplo).
    """
    bbox_min_idx = []
    bbox_max_idx = []
    for i, axis in enumerate(("y", "x", "z")):
        factor = downscale_factors[i]
        bbox_min_reduced = anchor_reduced[i] + offset_min_reduced[axis]
        bbox_max_reduced = anchor_reduced[i] + offset_max_reduced[axis]
        min_full = math.floor(bbox_min_reduced * factor)
        max_full = math.ceil(bbox_max_reduced * factor)
        min_full = max(0, min(min_full, volume_shape_full[i]))
        max_full = max(0, min(max_full, volume_shape_full[i]))
        if max_full <= min_full:
            raise ValueError(
                f"bbox vazio no eixo '{axis}' após clamp: [{min_full}, {max_full}) "
                f"(shape_full={volume_shape_full[i]})"
            )
        bbox_min_idx.append(min_full)
        bbox_max_idx.append(max_full)
    return (bbox_min_idx[0], bbox_min_idx[1], bbox_min_idx[2]), (
        bbox_max_idx[0], bbox_max_idx[1], bbox_max_idx[2]
    )


def crop_volume_and_update_affine(
    volume: NDArray[Any],
    affine: NDArray[Any],
    bbox_min: tuple[int, int, int],
    bbox_max: tuple[int, int, int],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Recorta ``volume`` pelos índices ``[bbox_min, bbox_max)`` (crop puro, sem interpolação) e
    devolve o affine correspondente.

    Nova origem = ``affine[:3, :3] @ offset_voxel + affine[:3, 3]`` — a fórmula geral de
    transformação afim, não a simplificação ``origin + offset * spacing`` (que só vale se a
    submatriz de rotação/escala do affine for diagonal **e positiva** em todos os eixos). Neste
    dataset ela **não é**: `nibabel.aff2axcodes` confirma orientação ``('L','A','S')`` e o eixo 0
    (y) tem entrada diagonal **negativa** (`affine[0,0] < 0`, ex. −0,377 mm no caso 1) — usar a
    fórmula simplificada erraria o sinal do deslocamento de origem nesse eixo. A fórmula geral é
    correta independente de sinal/orientação.
    """
    y0, x0, z0 = bbox_min
    y1, x1, z1 = bbox_max
    cropped = volume[y0:y1, x0:x1, z0:z1]

    new_affine = affine.copy()
    offset_vox = np.array([y0, x0, z0], dtype=float)
    new_affine[:3, 3] = affine[:3, :3] @ offset_vox + affine[:3, 3]
    return cropped, new_affine


def compute_real_label_loss(
    label_full: NDArray[Any], bbox_min: tuple[int, int, int], bbox_max: tuple[int, int, int]
) -> dict[str, Any]:
    """Perda real do label (voxel a voxel, resolução integral): fração dos voxels não-zero do
    label que caem **fora** de ``[bbox_min, bbox_max)``.

    Ao contrário do proxy geométrico de
    :func:`utils.segmentation.aorta_anchor.evaluate_percentile_bbox_tradeoff` (árvore = bloco
    retangular preenchendo sua caixa delimitadora) e do método de perda real em **resolução
    reduzida** já documentado (`docs/step6v2_aorta_anchor.md`, "Correção (2026-07-28/29)"), esta
    função opera diretamente na máscara em resolução integral, sem nenhum downscale — é a medida
    mais direta possível da perda real para um corte específico.
    """
    ys, xs, zs = np.nonzero(label_full)
    total = int(len(ys))
    if total == 0:
        raise ValueError("Label vazio (0 voxels não-zero) -- não é possível calcular perda.")

    y0, x0, z0 = bbox_min
    y1, x1, z1 = bbox_max
    inside = (
        (ys >= y0) & (ys < y1)
        & (xs >= x0) & (xs < x1)
        & (zs >= z0) & (zs < z1)
    )
    n_inside = int(inside.sum())
    return {
        "total_voxels": total,
        "n_inside": n_inside,
        "n_outside": total - n_inside,
        "loss_fraction": 1.0 - (n_inside / total),
    }


__all__ = [
    "compute_percentile_offsets",
    "compute_full_resolution_bbox",
    "crop_volume_and_update_affine",
    "compute_real_label_loss",
]
