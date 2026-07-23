"""Etapa 6 (v2) do bounding box cardíaco: âncora pela aorta, calibrada contra o label coronário.

Substitui a abordagem original de âncora (centróide do LCC, etapa 5/6 antigas) — descoberta na
sessão de 2026-07-15 como não seletiva o suficiente (isola o tronco inteiro, não o coração, ver
``docs/step6_envelope.md``). Aqui a âncora por caso é o centróide da máscara 3D da aorta ascendente
(Hough + Level Set, código herdado do pipeline de segmentação arterial de Silva et al., 2026,
módulos ``aorta_localization``/``aorta_segmentation``), e o tamanho fixo do bbox é calibrado
empiricamente a partir dos limites do label coronário em relação a essa âncora, no split de treino
— sem depender de "segmentar o coração" como etapa intermediária.

Determinístico ponta a ponta: a âncora vem de Hough+Level Set (processamento de imagem, sem
componente aprendido); os offsets calibrados são constantes fixas calculadas uma única vez a
partir do treino (agregação min/max, não um modelo); aplicar o bbox numa imagem nova usa só a
imagem (via a âncora) + os offsets já fixos, sem label.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..processing.binary_operations import keep_largest_component
from ..project.batch import run_resumable_batch
from .aorta_localization import detect_aorta_circles
from .aorta_segmentation import level_set_segmentation, remove_leaks_morphology
from .bounding_box import compute_bounding_box_bounds, compute_geometric_centroid
from .envelope import discover_dataset_blocks
from .pipeline_preprocessing import load_and_preprocess_image

AORTA_ANCHOR_RECORD_COLUMNS = [
    "img_id",
    "block",
    "status",
    "error_message",
    "anchor_y",
    "anchor_x",
    "anchor_z",
    "aorta_voxels",
    "num_circles_detected",
    "label_y_min",
    "label_y_max",
    "label_x_min",
    "label_x_max",
    "label_z_min",
    "label_z_max",
    "volume_shape_y",
    "volume_shape_x",
    "volume_shape_z",
    "processing_seconds",
]


def segment_aorta(
    lcc_image: NDArray[Any],
    scaled_spacing: tuple[float, float, float],
    config: dict[str, Any],
) -> tuple[NDArray[np.uint8], int]:
    """Localiza e segmenta a aorta ascendente (Hough + Level Set) num volume já pré-processado.

    Reaproveita ``detect_aorta_circles``/``level_set_segmentation`` como estão (código herdado do
    pipeline de segmentação arterial); não usa vesselness nem detecção de óstios — não são
    necessários para uma âncora posicional. Retorna máscara vazia (todos zeros) se nenhuma aorta
    for detectada; cabe ao chamador tratar isso como falha (``compute_geometric_centroid`` levanta
    ``ValueError`` para máscara vazia, que é o sinal de falha usado no processamento em lote).

    Returns:
        Tupla ``(mask, num_circles_detected)``.
    """
    circle_config = config["CIRCLE_DETECTION"]
    level_set_config = config["LEVEL_SET"]
    use_gpu = bool(config.get("USE_GPU", False))

    hough_radii = np.arange(
        circle_config["radii_start_px"],
        circle_config["radii_end_px"],
        circle_config.get("radius_step_px", 1),
    )
    # Espaçamento médio dos dois primeiros eixos (y, x) — usado só para converter mm<->pixel.
    pixel_spacing = (scaled_spacing[0] + scaled_spacing[1]) / 2.0

    detected_circles = detect_aorta_circles(
        lcc_image,
        hough_radii,
        pixel_spacing,
        tol_radius_mm=circle_config["tol_radius_mm"],
        tol_distance_mm=circle_config["tol_distance_mm"],
        quadrant_offset=tuple(circle_config["quadrant_offset"]),
        max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
        neighbor_distance_threshold=circle_config["neighbor_distance_threshold"],
        total_num_peaks_initial=circle_config["total_num_peaks_initial"],
        total_num_peaks=circle_config["total_num_peaks"],
        canny_sigma=circle_config["canny_sigma"],
        use_local_roi=circle_config.get("use_local_roi", True),
        local_roi_padding=circle_config.get("local_roi_padding", 20),
        out_of_tolerance_as_miss=circle_config.get("out_of_tolerance_as_miss", False),
        interpolate_missed_circles=circle_config.get("interpolate_missed_circles", True),
        candidate_selection_strategy=circle_config.get("candidate_selection_strategy", "closest"),
        candidate_score_accum_weight=circle_config.get("candidate_score_accum_weight", 1.0),
        candidate_score_distance_weight=circle_config.get(
            "candidate_score_distance_weight", 1.0
        ),
        candidate_score_radius_weight=circle_config.get("candidate_score_radius_weight", 1.0),
        use_gpu=use_gpu,
        verbose=False,
    )

    if not detected_circles:
        return np.zeros_like(lcc_image, dtype=np.uint8), 0

    mask = level_set_segmentation(
        lcc_image,
        detected_circles,
        radius_reduction_factor=level_set_config["radius_reduction_factor"],
        num_iter=level_set_config["num_iter"],
        balloon=level_set_config["balloon"],
        smoothing=level_set_config["smoothing"],
        use_gpu=use_gpu,
    )
    mask = remove_leaks_morphology(
        mask, radius=level_set_config["leak_removal_radius"], use_gpu=use_gpu
    )
    mask = keep_largest_component(mask, gpu=use_gpu)
    return mask.astype(np.uint8), len(detected_circles)


def process_case_for_aorta_anchor(
    img_id: int, base_path: str | Path, block: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Roda etapas 1-4 + detecção/segmentação da aorta e retorna a âncora + limites do label.

    Erros de um caso individual (arquivo ausente, aorta não detectada, label sem voxels
    coronários) ficam registrados com ``status="error"`` em vez de interromper o lote — dataset
    real de ~1000 casos com processamento caro (Hough + Level Set) tem falhas pontuais esperadas.
    """
    started_at = time.perf_counter()
    record: dict[str, Any] = {"img_id": img_id, "block": block}
    try:
        image_data = load_and_preprocess_image(img_id, str(base_path), config)
        lcc_image = image_data["lcc_image"]
        scaled_spacing = image_data["scaled_spacing"]
        label_artery = image_data["label"] == 1

        aorta_mask, num_circles = segment_aorta(lcc_image, scaled_spacing, config)
        # Levanta ValueError se a aorta não foi detectada (máscara vazia) — vira status="error".
        anchor = compute_geometric_centroid(aorta_mask)
        label_min, label_max = compute_bounding_box_bounds(label_artery)

        record.update(
            status="ok",
            error_message="",
            anchor_y=float(anchor[0]),
            anchor_x=float(anchor[1]),
            anchor_z=float(anchor[2]),
            aorta_voxels=int(aorta_mask.sum()),
            num_circles_detected=int(num_circles),
            label_y_min=int(label_min[0]),
            label_y_max=int(label_max[0]),
            label_x_min=int(label_min[1]),
            label_x_max=int(label_max[1]),
            label_z_min=int(label_min[2]),
            label_z_max=int(label_max[2]),
            volume_shape_y=int(lcc_image.shape[0]),
            volume_shape_x=int(lcc_image.shape[1]),
            volume_shape_z=int(lcc_image.shape[2]),
        )
    except Exception as exc:  # noqa: BLE001 - erro de caso real, deve seguir para o próximo
        record.update(
            status="error",
            error_message=str(exc),
            anchor_y=None,
            anchor_x=None,
            anchor_z=None,
            aorta_voxels=None,
            num_circles_detected=None,
            label_y_min=None,
            label_y_max=None,
            label_x_min=None,
            label_x_max=None,
            label_z_min=None,
            label_z_max=None,
            volume_shape_y=None,
            volume_shape_x=None,
            volume_shape_z=None,
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

    Mesmo padrão retomável de ``segmentation.envelope.process_block`` (via
    ``project.batch.run_resumable_batch``): pula casos ``status="ok"`` já salvos, sempre
    re-tenta casos com erro.
    """
    return run_resumable_batch(
        process_case_for_aorta_anchor,
        AORTA_ANCHOR_RECORD_COLUMNS,
        block,
        img_ids,
        base_path,
        config,
        output_csv,
        resume=resume,
        progress=progress,
    )


def run_aorta_anchor_batch(
    dataset_root: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    resume: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Processa todos os blocos descobertos em ``dataset_root`` e concatena os resultados.

    Mesmo padrão de ``segmentation.envelope.run_envelope_batch``: um CSV por bloco em
    ``output_dir`` (``block_<nome>.csv``), retomável de forma independente por bloco.
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


def calibrate_bbox_offsets(
    records_df: pd.DataFrame, downscale_factors: tuple[float, float, float]
) -> dict[str, Any]:
    """Calibra offsets fixos (bbox relativo à âncora) a partir de casos com label conhecido.

    ``records_df`` deve conter só os casos usados para calibração (tipicamente o split de
    treino, já filtrado pelo chamador) com ``status="ok"``. Para cada eixo, o offset mínimo é o
    mais conservador (mais negativo) entre ``label_min - anchor`` de todos os casos, e o offset
    máximo é o mais conservador (mais positivo) entre ``label_max - anchor`` — de modo que, para
    todo caso de calibração, ``anchor + offset_min <= label_min`` e ``label_max <= anchor +
    offset_max``. É uma agregação determinística (min/max), não um modelo treinado.
    """
    ok = records_df[records_df["status"] == "ok"]
    if ok.empty:
        raise ValueError(
            "Nenhum caso com status='ok' em records_df; não é possível calibrar os offsets."
        )

    offset_min: dict[str, float] = {}
    offset_max: dict[str, float] = {}
    argmin_case: dict[str, int] = {}
    argmax_case: dict[str, int] = {}
    for axis in ("y", "x", "z"):
        anchor_col = ok[f"anchor_{axis}"]
        low = ok[f"label_{axis}_min"] - anchor_col
        high = ok[f"label_{axis}_max"] - anchor_col
        offset_min[axis] = float(low.min())
        offset_max[axis] = float(high.max())
        argmin_case[axis] = int(ok.loc[low.idxmin(), "img_id"])
        argmax_case[axis] = int(ok.loc[high.idxmax(), "img_id"])

    offset_min_full = {
        axis: float(offset_min[axis] * factor)
        for axis, factor in zip(("y", "x", "z"), downscale_factors)
    }
    offset_max_full = {
        axis: float(offset_max[axis] * factor)
        for axis, factor in zip(("y", "x", "z"), downscale_factors)
    }
    bbox_extent_reduced = {axis: offset_max[axis] - offset_min[axis] for axis in ("y", "x", "z")}

    return {
        "num_cases_calibration": int(len(ok)),
        "downscale_factors": list(downscale_factors),
        "offset_min_reduced": offset_min,
        "offset_max_reduced": offset_max,
        "offset_min_full_resolution": offset_min_full,
        "offset_max_full_resolution": offset_max_full,
        "bbox_extent_reduced": bbox_extent_reduced,
        "argmin_case_per_axis": argmin_case,
        "argmax_case_per_axis": argmax_case,
    }


def calibrate_bbox_size_joint(
    records_df: pd.DataFrame,
    downscale_factors: tuple[float, float, float],
    percentile: float = 100.0,
) -> dict[str, Any]:
    """Calibra o bbox usando o extent 3D de UM caso de treino real, não uma combinação de
    extremos de casos diferentes por eixo (ver o achado documentado em
    ``docs/step6v2_aorta_anchor.md`` sobre ``calibrate_bbox_offsets``: agregar min/max
    independentemente por eixo combina o extremo de y de um paciente com o de x de outro e o
    de z de um terceiro — uma combinação que nenhum paciente real atinge, mas que infla o bbox
    muito além do necessário).

    Para cada caso, calcula seu próprio extent (``offset_max - offset_min`` por eixo) — um
    triplete ``(y, x, z)`` coerente, vindo do mesmo paciente. Ordena os casos pelo volume desse
    extent (voxels³, resolução reduzida) e usa o ``offset_min``/``offset_max`` do caso no
    ``percentile`` dessa ordenação como o bbox calibrado inteiro. O resultado corresponde
    sempre à anatomia de um paciente real observado no treino, nunca a uma combinação
    sintética.

    ``percentile=100`` (default) usa o caso de maior volume — o mais conservador possível
    dentro deste método, mas tipicamente ainda bem menor que ``calibrate_bbox_offsets``, já que
    não combina extremos de pacientes diferentes. Percentis menores usam um paciente de volume
    menor, aceitando que uma fração dos casos de treino fique parcialmente fora do bbox.

    O dict retornado usa as mesmas chaves de :func:`calibrate_bbox_offsets`
    (``offset_min_reduced``, ``offset_max_reduced``, ``bbox_extent_reduced``, ...) — compatível
    sem alteração com :func:`validate_coverage`.
    """
    ok = records_df[records_df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(
            "Nenhum caso com status='ok' em records_df; não é possível calibrar o bbox."
        )
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile deve estar em (0, 100], recebido {percentile}")

    for axis in ("y", "x", "z"):
        anchor_col = ok[f"anchor_{axis}"]
        ok[f"_offset_min_{axis}"] = ok[f"label_{axis}_min"] - anchor_col
        ok[f"_offset_max_{axis}"] = ok[f"label_{axis}_max"] - anchor_col
        ok[f"_extent_{axis}"] = ok[f"_offset_max_{axis}"] - ok[f"_offset_min_{axis}"]

    ok["_volume"] = ok["_extent_y"] * ok["_extent_x"] * ok["_extent_z"]
    sorted_ok = ok.sort_values("_volume").reset_index(drop=True)

    # Seleção por rank (não por interpolação de np.percentile): garante que o resultado seja
    # sempre o triplete de um caso real, nunca um valor sintético entre dois casos.
    rank = int(np.ceil(percentile / 100.0 * len(sorted_ok))) - 1
    rank = min(max(rank, 0), len(sorted_ok) - 1)
    reference = sorted_ok.iloc[rank]

    offset_min = {axis: float(reference[f"_offset_min_{axis}"]) for axis in ("y", "x", "z")}
    offset_max = {axis: float(reference[f"_offset_max_{axis}"]) for axis in ("y", "x", "z")}
    bbox_extent_reduced = {axis: offset_max[axis] - offset_min[axis] for axis in ("y", "x", "z")}

    offset_min_full = {
        axis: float(offset_min[axis] * factor)
        for axis, factor in zip(("y", "x", "z"), downscale_factors)
    }
    offset_max_full = {
        axis: float(offset_max[axis] * factor)
        for axis, factor in zip(("y", "x", "z"), downscale_factors)
    }

    return {
        "method": "joint_per_patient",
        "percentile": float(percentile),
        "num_cases_calibration": int(len(ok)),
        "reference_case_img_id": int(reference["img_id"]),
        "downscale_factors": list(downscale_factors),
        "offset_min_reduced": offset_min,
        "offset_max_reduced": offset_max,
        "offset_min_full_resolution": offset_min_full,
        "offset_max_full_resolution": offset_max_full,
        "bbox_extent_reduced": bbox_extent_reduced,
    }


def compute_per_case_envelope_size(
    records_df: pd.DataFrame, downscale_factors: tuple[float, float, float]
) -> pd.DataFrame:
    """Calcula, por caso e por eixo, o tamanho de envelope que aquele caso exigiria individualmente.

    Pré-requisito das tarefas seguintes da divisão combinada com o orientador (reunião
    2026-07-22: critérios de exclusão A e B, grade de varredura). Expõe por caso o mesmo cálculo
    de offset que :func:`calibrate_bbox_offsets` já faz internamente antes de agregar (min/max)
    entre casos — aqui sem agregação, um valor por caso. Não reprocessa nada: só lê as colunas já
    presentes em ``records_df`` (tipicamente ``all_cases.csv``).

    ``records_df`` deve estar já filtrado para os casos desejados (tipicamente ``status="ok"``) —
    esta função não filtra por status.

    Para cada eixo (y, x, z):

    - ``offset_min_<eixo>`` = ``label_<eixo>_min - anchor_<eixo>`` (mesmo sinal/convenção de
      :func:`calibrate_bbox_offsets`).
    - ``offset_max_<eixo>`` = ``label_<eixo>_max - anchor_<eixo>``.
    - ``envelope_extent_<eixo>`` = ``offset_max_<eixo> - offset_min_<eixo>`` — o tamanho que o bbox
      precisaria ter, centrado na âncora, para cobrir o label daquele caso especificamente
      (resolução reduzida).
    - ``envelope_extent_full_res_<eixo>`` = ``envelope_extent_<eixo> * downscale_factors[eixo]``
      (fator 1 no eixo z, já que o downscale preserva o eixo axial).

    Returns:
        DataFrame com ``img_id`` (int) e as 4 colunas acima por eixo, uma linha por caso de
        ``records_df`` (mesma ordem).

    Raises:
        ValueError: se ``records_df`` não tiver alguma das colunas necessárias (``img_id``,
            ``anchor_{y,x,z}``, ``label_{y,x,z}_{min,max}``) — sem fallback silencioso.
    """
    required_columns = {"img_id"}
    for axis in ("y", "x", "z"):
        required_columns.add(f"anchor_{axis}")
        required_columns.add(f"label_{axis}_min")
        required_columns.add(f"label_{axis}_max")

    missing = sorted(required_columns - set(records_df.columns))
    if missing:
        raise ValueError(
            "records_df não tem as colunas necessárias para "
            f"compute_per_case_envelope_size: {missing}"
        )

    result = pd.DataFrame({"img_id": records_df["img_id"].astype(int).to_numpy()})

    for axis, factor in zip(("y", "x", "z"), downscale_factors):
        offset_min = records_df[f"label_{axis}_min"] - records_df[f"anchor_{axis}"]
        offset_max = records_df[f"label_{axis}_max"] - records_df[f"anchor_{axis}"]
        envelope_extent = offset_max - offset_min

        result[f"offset_min_{axis}"] = offset_min.to_numpy(dtype=float)
        result[f"offset_max_{axis}"] = offset_max.to_numpy(dtype=float)
        result[f"envelope_extent_{axis}"] = envelope_extent.to_numpy(dtype=float)
        result[f"envelope_extent_full_res_{axis}"] = (envelope_extent * factor).to_numpy(
            dtype=float
        )

    return result


def validate_coverage(records_df: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    """Verifica, por caso, se o bbox calibrado (âncora + offsets fixos) cobre o label coronário.

    Pensado para rodar sobre casos **fora** da calibração (val/test) — ``calibration`` deve vir
    de :func:`calibrate_bbox_offsets` aplicada só ao split de treino, para uma checagem honesta
    de generalização (sem vazamento entre calibração e validação).
    """
    ok = records_df[records_df["status"] == "ok"].copy()
    offset_min = calibration["offset_min_reduced"]
    offset_max = calibration["offset_max_reduced"]

    for axis in ("y", "x", "z"):
        anchor_col = ok[f"anchor_{axis}"]
        ok[f"bbox_{axis}_min"] = anchor_col + offset_min[axis]
        ok[f"bbox_{axis}_max"] = anchor_col + offset_max[axis]
        ok[f"covered_{axis}"] = (ok[f"label_{axis}_min"] >= ok[f"bbox_{axis}_min"]) & (
            ok[f"label_{axis}_max"] <= ok[f"bbox_{axis}_max"]
        )

    ok["covered"] = ok["covered_y"] & ok["covered_x"] & ok["covered_z"]
    return ok


__all__ = [
    "AORTA_ANCHOR_RECORD_COLUMNS",
    "segment_aorta",
    "process_case_for_aorta_anchor",
    "process_block",
    "run_aorta_anchor_batch",
    "calibrate_bbox_offsets",
    "calibrate_bbox_size_joint",
    "compute_per_case_envelope_size",
    "validate_coverage",
]
