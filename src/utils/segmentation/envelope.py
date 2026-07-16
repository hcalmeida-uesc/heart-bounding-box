"""Etapa 6 do bounding box cardíaco: envelope de dimensões máximas sobre o dataset completo.

Processa os casos do ImageCAS em blocos que coincidem com a divisão de pastas do dataset local
(``dataset/1-200``, ``dataset/201-400``, ...), persistindo centróide e extensão do bbox mínimo por
caso de forma retomável (``process_block`` pula casos já processados com sucesso), para depois
derivar o envelope conservador (dimensão fixa) usado nas etapas 7-8 (``compute_envelope``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..project.batch import run_resumable_batch
from .bounding_box import compute_bounding_extent, compute_geometric_centroid
from .pipeline_preprocessing import load_and_preprocess_image

ENVELOPE_RECORD_COLUMNS = [
    "img_id",
    "block",
    "status",
    "error_message",
    "centroid_y",
    "centroid_x",
    "centroid_z",
    "extent_y",
    "extent_x",
    "extent_z",
    "volume_shape_y",
    "volume_shape_x",
    "volume_shape_z",
    "lcc_voxels",
    "processing_seconds",
]


def discover_dataset_blocks(dataset_root: str | Path) -> dict[str, list[int]]:
    """Lista os blocos (subpastas) do dataset local e os ``img_id`` de cada um.

    Cada subpasta imediata de ``dataset_root`` que contenha arquivos ``*.img.nii.gz`` vira um
    bloco, na convenção já usada por ``check_preprocessing_and_centroid.ipynb`` (faixas de 200
    IDs: ``1-200``, ``201-400``, ...). Não assume nomes de pasta fixos — descobre dinamicamente.
    """
    dataset_root = Path(dataset_root)
    blocks: dict[str, list[int]] = {}
    for block_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        img_paths = sorted(
            block_dir.glob("*.img.nii.gz"),
            key=lambda path: int(path.name.split(".")[0]),
        )
        if img_paths:
            blocks[block_dir.name] = [int(path.name.split(".")[0]) for path in img_paths]

    if not blocks:
        raise FileNotFoundError(
            f"Nenhum bloco com *.img.nii.gz encontrado em {dataset_root}"
        )
    return blocks


def process_case_for_envelope(
    img_id: int, base_path: str | Path, block: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Roda as etapas 1-5 para um caso e retorna centróide + extensão do bbox mínimo (etapa 6).

    Erros de um caso individual (arquivo ausente/corrompido, máscara vazia após threshold/LCC
    etc.) não interrompem o processamento em lote — ficam registrados no record com
    ``status="error"`` para inspeção posterior, já que este processa dados reais do dataset e
    exceções de I/O/dados são esperadas em uma execução de ~1000 casos, não bugs de código.
    """
    started_at = time.perf_counter()
    record: dict[str, Any] = {"img_id": img_id, "block": block}
    try:
        image_data = load_and_preprocess_image(img_id, str(base_path), config)
        lcc_mask = image_data["lcc_mask"]
        centroid = compute_geometric_centroid(lcc_mask)
        extent = compute_bounding_extent(lcc_mask)
        record.update(
            status="ok",
            error_message="",
            centroid_y=float(centroid[0]),
            centroid_x=float(centroid[1]),
            centroid_z=float(centroid[2]),
            extent_y=int(extent[0]),
            extent_x=int(extent[1]),
            extent_z=int(extent[2]),
            volume_shape_y=int(lcc_mask.shape[0]),
            volume_shape_x=int(lcc_mask.shape[1]),
            volume_shape_z=int(lcc_mask.shape[2]),
            lcc_voxels=int(image_data["preprocessing_details"]["lcc_voxels"]),
        )
    except Exception as exc:  # noqa: BLE001 - erro de caso real, deve seguir para o próximo
        record.update(
            status="error",
            error_message=str(exc),
            centroid_y=None,
            centroid_x=None,
            centroid_z=None,
            extent_y=None,
            extent_x=None,
            extent_z=None,
            volume_shape_y=None,
            volume_shape_x=None,
            volume_shape_z=None,
            lcc_voxels=None,
        )
    record["processing_seconds"] = time.perf_counter() - started_at
    return record


def process_block(
    block: str,
    img_ids: list[int],
    base_path: str | Path,
    config: dict[str, Any],
    output_csv: str | Path,
    resume: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Processa um bloco (pasta) inteiro, salvando incrementalmente em ``output_csv``.

    Casos já registrados com ``status="ok"`` em uma execução anterior são pulados quando
    ``resume=True`` (default) — permite retomar após interrupção sem reprocessar o bloco inteiro.
    Casos com erro são sempre re-tentados. Delega o checkpoint incremental a
    ``project.batch.run_resumable_batch``, compartilhado com outras etapas do bounding box.
    """
    return run_resumable_batch(
        process_case_for_envelope,
        ENVELOPE_RECORD_COLUMNS,
        block,
        img_ids,
        base_path,
        config,
        output_csv,
        resume=resume,
        progress=progress,
    )


def run_envelope_batch(
    dataset_root: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    resume: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Processa todos os blocos descobertos em ``dataset_root`` e concatena os resultados.

    Salva um CSV por bloco em ``output_dir`` (``block_<nome>.csv``), cada um retomável de forma
    independente — se a execução for interrompida, rodar de novo pula os blocos/casos já com
    ``status="ok"``.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    blocks = discover_dataset_blocks(dataset_root)

    all_records = []
    for block, img_ids in blocks.items():
        base_path = dataset_root / block
        output_csv = output_dir / f"block_{block}.csv"
        block_df = process_block(
            block, img_ids, base_path, config, output_csv, resume=resume, progress=progress
        )
        all_records.append(block_df)

    return pd.concat(all_records, ignore_index=True).sort_values("img_id").reset_index(drop=True)


def compute_envelope(
    records_df: pd.DataFrame, downscale_factors: tuple[float, float, float]
) -> dict[str, Any]:
    """Deriva o envelope de dimensões fixas (etapa 6) a partir dos extents por caso.

    Usa o maior extent observado em cada eixo (na resolução reduzida em que os casos foram
    processados) e escala de volta para a resolução integral multiplicando por
    ``downscale_factors`` — alinhado eixo a eixo com a convenção ``(Y, X, Z)`` do resto do
    pipeline (fator 1 no eixo z, já que o downscale preserva o eixo axial).
    """
    ok = records_df[records_df["status"] == "ok"]
    if ok.empty:
        raise ValueError(
            "Nenhum caso com status='ok' em records_df; não é possível calcular o envelope."
        )

    extent_cols = ["extent_y", "extent_x", "extent_z"]
    extent_reduced = ok[extent_cols].astype(int).max()

    argmax_case_per_axis = {
        axis: int(ok.loc[ok[f"extent_{axis}"].astype(int).idxmax(), "img_id"])
        for axis in ("y", "x", "z")
    }

    extent_full_resolution = {
        axis: int(extent_reduced[f"extent_{axis}"] * factor)
        for axis, factor in zip(("y", "x", "z"), downscale_factors)
    }

    return {
        "num_cases_ok": int(len(ok)),
        "num_cases_error": int(len(records_df) - len(ok)),
        "downscale_factors": list(downscale_factors),
        "extent_reduced": {
            axis: int(extent_reduced[f"extent_{axis}"]) for axis in ("y", "x", "z")
        },
        "extent_full_resolution": extent_full_resolution,
        "argmax_case_per_axis": argmax_case_per_axis,
    }


__all__ = [
    "ENVELOPE_RECORD_COLUMNS",
    "discover_dataset_blocks",
    "process_case_for_envelope",
    "process_block",
    "run_envelope_batch",
    "compute_envelope",
]
