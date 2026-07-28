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


def compute_per_case_envelope_size(
    records_df: pd.DataFrame, downscale_factors: tuple[float, float, float]
) -> pd.DataFrame:
    """Calcula, por caso e por eixo, o tamanho de envelope que aquele caso exigiria individualmente.

    Mesmo cálculo de base que :func:`calibrate_bbox_offsets` já faz internamente por eixo
    (``offset = label - anchor``), mas aqui expondo o resultado **por caso**, não agregado
    (min/max) entre casos — preparação para os critérios de exclusão da próxima etapa (tarefa
    2/7). Não reprocessa nada: opera só sobre colunas já presentes em ``records_df``
    (tipicamente ``all_cases.csv``), filtrado internamente para ``status="ok"``.

    Para cada eixo, retorna ``offset_min``/``offset_max`` (``label_min/max - anchor``, mesma
    convenção de sinal de :func:`calibrate_bbox_offsets`), ``envelope_extent`` (``offset_max -
    offset_min``, resolução reduzida) e ``envelope_extent_full_res`` (escalado por
    ``downscale_factors``, fator 1 no eixo z).

    Nota (validação manual, 2026-07-28): ``img_id=924`` é o caso de maior ``envelope_extent_z``
    entre os 994 casos processados (247, resolução reduzida) — o mesmo caso já identificado em
    ``docs/step6v2_aorta_anchor.md`` (diagnóstico de 2026-07-24/25) como pior caso de excesso de
    cobertura em z na produção (under-tracking, só 2 círculos detectados pelo Hough). Duas
    métricas independentes (excesso do bbox calibrado vs. tamanho de envelope por caso, sem
    depender de calibração alguma) concordam no mesmo caso — referência direta para a comparação
    dos critérios de exclusão A/B na tarefa 5/7.
    """
    required_columns = {
        "img_id",
        "status",
        "anchor_y",
        "anchor_x",
        "anchor_z",
        "label_y_min",
        "label_y_max",
        "label_x_min",
        "label_x_max",
        "label_z_min",
        "label_z_max",
    }
    missing = required_columns - set(records_df.columns)
    if missing:
        raise ValueError(
            f"records_df não contém as colunas esperadas: {sorted(missing)}"
        )

    ok = records_df[records_df["status"] == "ok"]
    if ok.empty:
        raise ValueError(
            "Nenhum caso com status='ok' em records_df; não é possível calcular o envelope."
        )

    result = ok[["img_id"]].copy()
    for axis, factor in zip(("y", "x", "z"), downscale_factors):
        anchor_col = ok[f"anchor_{axis}"]
        offset_min = ok[f"label_{axis}_min"] - anchor_col
        offset_max = ok[f"label_{axis}_max"] - anchor_col
        envelope_extent = offset_max - offset_min

        result[f"offset_min_{axis}"] = offset_min.astype(float)
        result[f"offset_max_{axis}"] = offset_max.astype(float)
        result[f"envelope_extent_{axis}"] = envelope_extent.astype(float)
        result[f"envelope_extent_full_res_{axis}"] = (envelope_extent * factor).astype(float)

    return result.reset_index(drop=True)


def select_cases_by_circle_count_percentile(
    records_df: pd.DataFrame, exclusion_pct: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Critério de exclusão A: remove os casos nas duas caudas da distribuição de
    ``num_circles_detected``.

    Tentativa de excluir outliers de rastreamento Hough+Level Set (under-/over-tracking) na
    calibração, motivada pelo padrão em U encontrado no diagnóstico de cobertura em z (ver
    ``docs/step6v2_aorta_anchor.md``, achado por quintil de ``num_circles_detected``: falha de
    cobertura em z concentrada nos dois extremos da distribuição, zero nas faixas centrais).

    ``exclusion_pct`` é o percentual **total** de exclusão, dividido igualmente entre as duas
    caudas (ex. ``exclusion_pct=5.0`` remove 2,5% do lado de baixo + 2,5% do lado de cima). Usa
    percentil (``numpy.percentile``, interpolação linear) sobre ``num_circles_detected`` dos
    casos com ``status="ok"`` — não uma contagem fixa de casos — para que ``exclusion_pct``
    fracionário funcione corretamente. Um caso é excluído se seu ``num_circles_detected`` for
    estritamente menor que o percentil inferior ou estritamente maior que o percentil superior;
    com ``exclusion_pct=0`` os dois percentis viram o mínimo/máximo exatos da distribuição e a
    comparação estrita não exclui nenhum caso.

    Retorna uma tupla ``(excluded_df, filtered_df)``:

    - ``excluded_df``: um caso por linha (``img_id``, ``side`` = ``"baixo"``/``"alto"``,
      ``num_circles_detected``), ordenado por ``img_id``.
    - ``filtered_df``: os casos ``status="ok"`` restantes (fora das duas caudas), mesmas colunas
      de ``records_df``.
    """
    if not (0 <= exclusion_pct < 100):
        raise ValueError(f"exclusion_pct deve estar em [0, 100), recebido {exclusion_pct}")

    required_columns = {"img_id", "status", "num_circles_detected"}
    missing = required_columns - set(records_df.columns)
    if missing:
        raise ValueError(
            f"records_df não contém as colunas esperadas: {sorted(missing)}"
        )

    ok = records_df[records_df["status"] == "ok"]
    if ok.empty:
        raise ValueError(
            "Nenhum caso com status='ok' em records_df; não é possível selecionar por percentil."
        )

    tail_pct = exclusion_pct / 2.0
    low_threshold = float(np.percentile(ok["num_circles_detected"], tail_pct))
    high_threshold = float(np.percentile(ok["num_circles_detected"], 100.0 - tail_pct))

    is_low = ok["num_circles_detected"] < low_threshold
    is_high = ok["num_circles_detected"] > high_threshold

    excluded_low = ok.loc[is_low, ["img_id", "num_circles_detected"]].assign(side="baixo")
    excluded_high = ok.loc[is_high, ["img_id", "num_circles_detected"]].assign(side="alto")
    excluded_df = (
        pd.concat([excluded_low, excluded_high], ignore_index=True)
        .sort_values("img_id")
        .reset_index(drop=True)[["img_id", "side", "num_circles_detected"]]
    )

    filtered_df = ok.loc[~(is_low | is_high)].reset_index(drop=True)

    return excluded_df, filtered_df


def select_cases_by_envelope_size_percentile(
    envelope_df: pd.DataFrame, exclusion_pct: float, axis: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Critério de exclusão B: remove os casos nas duas caudas da distribuição de
    ``envelope_extent_{axis}``, um eixo por vez.

    Independente do critério A (:func:`select_cases_by_circle_count_percentile`) — os dois serão
    comparados lado a lado, não combinados, na próxima etapa (tarefa 5/7). Recebe o DataFrame já
    calculado por :func:`compute_per_case_envelope_size` (não reprocessa nada).

    Mesma lógica de divisão de caudas da tarefa 2/7, para manter os dois critérios comparáveis no
    mesmo desenho: ``exclusion_pct`` é o percentual **total** de exclusão, dividido igualmente
    entre as duas caudas (ex. ``exclusion_pct=5.0`` remove 2,5% do lado de baixo + 2,5% do lado de
    cima) via percentil (``numpy.percentile``, interpolação linear) sobre
    ``envelope_extent_{axis}``. Um caso é excluído se seu valor for estritamente menor que o
    percentil inferior ou estritamente maior que o percentil superior; com ``exclusion_pct=0`` os
    dois percentis viram o mínimo/máximo exatos e a comparação estrita não exclui nenhum caso.

    Retorna uma tupla ``(excluded_df, filtered_df)``:

    - ``excluded_df``: um caso por linha (``img_id``, ``side`` = ``"baixo"``/``"alto"``,
      ``envelope_extent``), ordenado por ``img_id``.
    - ``filtered_df``: os casos restantes (fora das duas caudas), mesmas colunas de
      ``envelope_df``.

    **Reclassificação metodológica (tarefa 3b/7, 2026-07-28): este critério usa
    ``label_min``/``label_max`` do próprio caso (via ``envelope_extent``, calculado por
    :func:`compute_per_case_envelope_size`) — ou seja, depende do label coronário, que só existe
    porque o ImageCAS é um dataset rotulado.** Por isso só serve como critério de **curadoria de
    dataset** — decidir quais dos 1000 casos já rotulados entram no estudo, análogo a critérios de
    inclusão/exclusão de qualquer benchmark — e **não pode ser parte do pipeline determinístico**
    aplicado a uma imagem nova sem anotação (que é o objetivo do Cap. 2). Ver
    :func:`select_cases_by_aorta_volume_percentile` (critério C) para um contraponto derivável só
    da imagem, sem label, elegível para produção.
    """
    if axis not in ("y", "x", "z"):
        raise ValueError(f"axis deve ser 'y', 'x' ou 'z', recebido {axis!r}")
    if not (0 <= exclusion_pct < 100):
        raise ValueError(f"exclusion_pct deve estar em [0, 100), recebido {exclusion_pct}")

    extent_col = f"envelope_extent_{axis}"
    required_columns = {"img_id", extent_col}
    missing = required_columns - set(envelope_df.columns)
    if missing:
        raise ValueError(
            f"envelope_df não contém as colunas esperadas: {sorted(missing)}"
        )

    tail_pct = exclusion_pct / 2.0
    low_threshold = float(np.percentile(envelope_df[extent_col], tail_pct))
    high_threshold = float(np.percentile(envelope_df[extent_col], 100.0 - tail_pct))

    is_low = envelope_df[extent_col] < low_threshold
    is_high = envelope_df[extent_col] > high_threshold

    excluded_low = (
        envelope_df.loc[is_low, ["img_id", extent_col]]
        .rename(columns={extent_col: "envelope_extent"})
        .assign(side="baixo")
    )
    excluded_high = (
        envelope_df.loc[is_high, ["img_id", extent_col]]
        .rename(columns={extent_col: "envelope_extent"})
        .assign(side="alto")
    )
    excluded_df = (
        pd.concat([excluded_low, excluded_high], ignore_index=True)
        .sort_values("img_id")
        .reset_index(drop=True)[["img_id", "side", "envelope_extent"]]
    )

    filtered_df = envelope_df.loc[~(is_low | is_high)].reset_index(drop=True)

    return excluded_df, filtered_df


def select_cases_by_aorta_volume_percentile(
    records_df: pd.DataFrame, exclusion_pct: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Critério de exclusão C: remove os casos nas duas caudas da distribuição de
    ``aorta_voxels`` (volume da máscara da aorta segmentada).

    Mesmo desenho de :func:`select_cases_by_circle_count_percentile` (critério A), trocando
    ``num_circles_detected`` por ``aorta_voxels``. Ao contrário do critério B
    (:func:`select_cases_by_envelope_size_percentile`), **não depende do label coronário** — é
    derivável só da imagem (a máscara da aorta vem de ``segment_aorta``, sem nenhuma anotação) —
    por isso é elegível para uso no pipeline determinístico de produção, não só como critério de
    curadoria de dataset. Motivado pelo mesmo achado que originou o critério A: os casos de pior
    cobertura em z tinham volume de máscara muito distante da referência (``docs/step6v2_aorta_
    anchor.md``, diagnóstico de 2026-07-25, "quantificação do leak") — under-tracking (máscara
    minúscula) e over-tracking/leak (máscara enorme) são mecanismos distintos, ambos refletidos no
    volume total da máscara.

    ``exclusion_pct`` é o percentual **total** de exclusão, dividido igualmente entre as duas
    caudas (ex. ``exclusion_pct=5.0`` remove 2,5% do lado de baixo + 2,5% do lado de cima) via
    percentil (``numpy.percentile``, interpolação linear) sobre ``aorta_voxels`` dos casos com
    ``status="ok"``. Um caso é excluído se seu ``aorta_voxels`` for estritamente menor que o
    percentil inferior ou estritamente maior que o percentil superior; com ``exclusion_pct=0`` os
    dois percentis viram o mínimo/máximo exatos da distribuição e a comparação estrita não exclui
    nenhum caso. Não reprocessa nada: ``aorta_voxels`` já está persistido em ``all_cases.csv``
    para os 994 casos.

    Retorna uma tupla ``(excluded_df, filtered_df)``:

    - ``excluded_df``: um caso por linha (``img_id``, ``side`` = ``"baixo"``/``"alto"``,
      ``aorta_voxels``), ordenado por ``img_id``.
    - ``filtered_df``: os casos ``status="ok"`` restantes (fora das duas caudas), mesmas colunas
      de ``records_df``.
    """
    if not (0 <= exclusion_pct < 100):
        raise ValueError(f"exclusion_pct deve estar em [0, 100), recebido {exclusion_pct}")

    required_columns = {"img_id", "status", "aorta_voxels"}
    missing = required_columns - set(records_df.columns)
    if missing:
        raise ValueError(
            f"records_df não contém as colunas esperadas: {sorted(missing)}"
        )

    ok = records_df[records_df["status"] == "ok"]
    if ok.empty:
        raise ValueError(
            "Nenhum caso com status='ok' em records_df; não é possível selecionar por percentil."
        )

    tail_pct = exclusion_pct / 2.0
    low_threshold = float(np.percentile(ok["aorta_voxels"], tail_pct))
    high_threshold = float(np.percentile(ok["aorta_voxels"], 100.0 - tail_pct))

    is_low = ok["aorta_voxels"] < low_threshold
    is_high = ok["aorta_voxels"] > high_threshold

    excluded_low = ok.loc[is_low, ["img_id", "aorta_voxels"]].assign(side="baixo")
    excluded_high = ok.loc[is_high, ["img_id", "aorta_voxels"]].assign(side="alto")
    excluded_df = (
        pd.concat([excluded_low, excluded_high], ignore_index=True)
        .sort_values("img_id")
        .reset_index(drop=True)[["img_id", "side", "aorta_voxels"]]
    )

    filtered_df = ok.loc[~(is_low | is_high)].reset_index(drop=True)

    return excluded_df, filtered_df


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


def _filter_splits_by_train_threshold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    criterion_column: str,
    exclusion_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """Calcula o limiar de corte (percentil inferior/superior de ``criterion_column``) só com
    ``train_df`` e aplica o valor absoluto resultante aos três splits.

    Helper compartilhado por :func:`evaluate_exclusion_criterion` (tarefa 4/7) e
    :func:`evaluate_percentile_bbox_tradeoff` (esclarecimento pós-reunião) — mesma lógica de corte
    em ambos, para que os dois jeitos de usar um critério de exclusão (recalibração min/max vs.
    bbox por percentil) sejam comparáveis.
    """
    tail_pct = exclusion_pct / 2.0
    low_threshold = float(np.percentile(train_df[criterion_column], tail_pct))
    high_threshold = float(np.percentile(train_df[criterion_column], 100.0 - tail_pct))

    def _filter_by_threshold(split_df: pd.DataFrame) -> pd.DataFrame:
        in_range = (split_df[criterion_column] >= low_threshold) & (
            split_df[criterion_column] <= high_threshold
        )
        return split_df.loc[in_range]

    return (
        _filter_by_threshold(train_df),
        _filter_by_threshold(val_df),
        _filter_by_threshold(test_df),
        low_threshold,
        high_threshold,
    )


def _raise_if_any_split_empty(
    train_filtered: pd.DataFrame,
    val_filtered: pd.DataFrame,
    test_filtered: pd.DataFrame,
    criterion_column: str,
    exclusion_pct: float,
    low_threshold: float,
    high_threshold: float,
) -> None:
    for split_name, split_filtered in (
        ("treino", train_filtered),
        ("val", val_filtered),
        ("test", test_filtered),
    ):
        if split_filtered.empty:
            raise ValueError(
                f"Split '{split_name}' ficou vazio após o filtro por '{criterion_column}' "
                f"(exclusion_pct={exclusion_pct}, low={low_threshold}, high={high_threshold})."
            )


def evaluate_exclusion_criterion(
    records_df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    criterion_column: str,
    exclusion_pct: float,
    downscale_factors: tuple[float, float, float],
) -> dict[str, Any]:
    """Recalibra o bbox de tamanho fixo e valida cobertura sobre um subconjunto filtrado por um
    critério de exclusão (tarefa 4/7).

    O limiar de corte (percentil inferior/superior de ``criterion_column``, ex.
    ``num_circles_detected`` do critério A ou ``aorta_voxels`` do critério C — **não usar o
    critério B/``envelope_extent``**, reclassificado como curadoria de dataset, não elegível para
    pipeline) é calculado **só com os casos de treino** — do mesmo jeito que
    :func:`calibrate_bbox_offsets` já usa só o treino para calibrar offsets. O valor absoluto
    resultante (não um novo percentil) é aplicado igualmente aos três splits, para que val/test
    sejam filtrados pelo mesmo critério que definiu o corte, sem vazamento de informação de
    val/test para a definição do limiar.

    Não reprocessa nada: opera só sobre colunas já presentes em ``records_df`` (tipicamente
    ``all_cases.csv``) — inclusive o `Rv` (redução volumétrica), aproximado em **contagem de
    voxels** a partir de ``volume_shape_{y,x,z}`` (shape do volume já reduzido, escalado de volta
    para resolução integral via ``downscale_factors``) — não o volume físico (mm³) real, que
    exigiria reler o espaçamento de cada NIfTI (fora do escopo desta tarefa).

    Casos de val/test excluídos pelo critério ficam **fora de escopo** — não contam nem como
    cobertos nem como falha; o retorno separa ``n_val_excluded``/``n_test_excluded`` (fora de
    escopo) de ``coverage_n_evaluated`` (só os casos incluídos), para não confundir "cobertura
    melhorou" com "avaliamos menos casos".

    Levanta ``ValueError`` se ``exclusion_pct`` estiver fora de ``[0, 100)``, se faltar alguma
    coluna necessária, ou se treino, val ou test ficarem vazios **depois** do filtro (calibração/
    validação não fazem sentido sobre um split vazio).

    Retorna um dict com: limiares (``low_threshold``, ``high_threshold``), N total/excluído/
    incluído por split, a calibração completa (mesmo dict de :func:`calibrate_bbox_offsets`),
    ``bbox_extent_reduced``/``bbox_extent_full_resolution`` por eixo, estatísticas de `Rv` (`rv_
    mean`, `rv_median`, `rv_std`, `rv_min`, `rv_max`, sobre todos os casos incluídos dos 3 splits)
    e cobertura conjunta/por eixo sobre val+test incluídos.
    """
    if not (0 <= exclusion_pct < 100):
        raise ValueError(f"exclusion_pct deve estar em [0, 100), recebido {exclusion_pct}")

    required_columns = {
        "img_id",
        "status",
        criterion_column,
        "anchor_y",
        "anchor_x",
        "anchor_z",
        "label_y_min",
        "label_y_max",
        "label_x_min",
        "label_x_max",
        "label_z_min",
        "label_z_max",
        "volume_shape_y",
        "volume_shape_x",
        "volume_shape_z",
    }
    missing = required_columns - set(records_df.columns)
    if missing:
        raise ValueError(
            f"records_df não contém as colunas esperadas: {sorted(missing)}"
        )

    ok = records_df[records_df["status"] == "ok"]
    train_df = ok[ok["img_id"].isin(train_ids)]
    val_df = ok[ok["img_id"].isin(val_ids)]
    test_df = ok[ok["img_id"].isin(test_ids)]

    if train_df.empty:
        raise ValueError("Nenhum caso de treino com status='ok' em records_df.")

    train_filtered, val_filtered, test_filtered, low_threshold, high_threshold = (
        _filter_splits_by_train_threshold(
            train_df, val_df, test_df, criterion_column, exclusion_pct
        )
    )
    _raise_if_any_split_empty(
        train_filtered, val_filtered, test_filtered, criterion_column, exclusion_pct,
        low_threshold, high_threshold,
    )

    calibration = calibrate_bbox_offsets(train_filtered, downscale_factors)

    eval_df = pd.concat([val_filtered, test_filtered], ignore_index=True)
    coverage_df = validate_coverage(eval_df, calibration)

    bbox_extent_full_resolution = {
        axis: calibration["offset_max_full_resolution"][axis]
        - calibration["offset_min_full_resolution"][axis]
        for axis in ("y", "x", "z")
    }
    v_bbox_full_res = (
        bbox_extent_full_resolution["y"]
        * bbox_extent_full_resolution["x"]
        * bbox_extent_full_resolution["z"]
    )

    included_df = pd.concat([train_filtered, val_filtered, test_filtered], ignore_index=True)
    v_original_full_res = (
        (included_df["volume_shape_y"] * downscale_factors[0])
        * (included_df["volume_shape_x"] * downscale_factors[1])
        * (included_df["volume_shape_z"] * downscale_factors[2])
    )
    rv_per_case = 1.0 - (v_bbox_full_res / v_original_full_res)

    return {
        "criterion_column": criterion_column,
        "exclusion_pct": float(exclusion_pct),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "n_train_total": int(len(train_df)),
        "n_train_excluded": int(len(train_df) - len(train_filtered)),
        "n_train_included": int(len(train_filtered)),
        "n_val_total": int(len(val_df)),
        "n_val_excluded": int(len(val_df) - len(val_filtered)),
        "n_val_included": int(len(val_filtered)),
        "n_test_total": int(len(test_df)),
        "n_test_excluded": int(len(test_df) - len(test_filtered)),
        "n_test_included": int(len(test_filtered)),
        "calibration": calibration,
        "bbox_extent_reduced": calibration["bbox_extent_reduced"],
        "bbox_extent_full_resolution": bbox_extent_full_resolution,
        "rv_n_cases": int(len(included_df)),
        "rv_mean": float(rv_per_case.mean()),
        "rv_median": float(rv_per_case.median()),
        "rv_std": float(rv_per_case.std()),
        "rv_min": float(rv_per_case.min()),
        "rv_max": float(rv_per_case.max()),
        "coverage_n_evaluated": int(len(coverage_df)),
        "coverage_joint": float(coverage_df["covered"].mean()),
        "coverage_y": float(coverage_df["covered_y"].mean()),
        "coverage_x": float(coverage_df["covered_x"].mean()),
        "coverage_z": float(coverage_df["covered_z"].mean()),
    }


def evaluate_percentile_bbox_tradeoff(
    records_df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    offset_percentile: float,
    downscale_factors: tuple[float, float, float],
    exclude_via_criterion: str | None = None,
    exclusion_pct: float | None = None,
) -> dict[str, Any]:
    """Reteste (esclarecimento pós-reunião) da recalibração por percentil de 2026-07-17: em vez
    de excluir casos (tarefas 3b/4), mantém todos os casos do treino (filtrado ou não) e define o
    bbox por um percentil do offset em cada eixo — não o mínimo/máximo absoluto — medindo perda
    **parcial** (não mais um pass/fail binário) sobre val/test.

    Para cada eixo, ``offset_min`` vem do percentil ``100 - offset_percentile`` da distribuição
    ``label_min - âncora`` do treino, e ``offset_max`` vem do percentil ``offset_percentile`` da
    distribuição ``label_max - âncora`` — mesma convenção marginal por eixo de
    :func:`calibrate_bbox_offsets`, generalizada para qualquer nível. Com
    ``offset_percentile=100`` os dois percentis viram mínimo/máximo exatos — **reproduz
    :func:`calibrate_bbox_offsets` exatamente**, regressão de sanidade coberta em teste.

    Se ``exclude_via_criterion`` for fornecido (junto com ``exclusion_pct``), filtra primeiro os
    três splits pelo limiar calculado só no treino (mesma lógica de
    :func:`evaluate_exclusion_criterion`, via ``_filter_splits_by_train_threshold``) — combinando
    exclusão de casos (tarefa 4) com bbox por percentil (esta tarefa) no mesmo experimento. Sem
    ``exclude_via_criterion``, nenhum caso é excluído (reteste direto do experimento de
    2026-07-17, agora com perda parcial em vez de só cobertura conjunta/`Rv`).

    **Aproximação usada para a perda parcial (importante, não é a máscara real):** como
    ``records_df`` não guarda a máscara voxel a voxel do label coronário (só
    ``label_{y,x,z}_min/max``, os limites da caixa delimitadora), a árvore coronária é
    aproximada aqui como um **bloco retangular preenchendo sua própria caixa delimitadora** — a
    mesma aproximação que :func:`validate_coverage` já usa para o teste binário, só que estendida
    para uma fração contínua: ``perda = 1 - (volume da interseção entre a caixa do label e o bbox
    calibrado) / (volume da caixa do label)``. É 0 exatamente quando o label cabe inteiro no bbox
    (mesmo caso em que ``validate_coverage`` marcaria ``covered=True``) e cresce conforme o bbox
    encolhe abaixo da extensão do label em algum eixo. **Não é a fração real de voxels da árvore
    perdidos** (a árvore não preenche sua caixa delimitadora — é uma estrutura fina e ramificada,
    não um bloco sólido) — é um proxy geométrico, consistente com o resto do módulo (que nunca
    reprocessa a máscara), não uma medição exata.

    Levanta ``ValueError`` se ``offset_percentile`` estiver fora de ``(50, 100]``, se
    ``exclude_via_criterion``/``exclusion_pct`` forem fornecidos só um dos dois, se faltar alguma
    coluna necessária, ou se o treino ficar vazio (ou, quando ``exclude_via_criterion`` é usado,
    se algum split ficar vazio depois do filtro).

    Retorna um dict com: contagens por split (total/excluído/incluído, iguais em todos os campos
    ``_excluded`` quando ``exclude_via_criterion=None``), ``bbox_extent_reduced``/
    ``bbox_extent_full_resolution`` por eixo, `Rv` (proxy de voxels, `rv_mean`/`rv_median`) e a
    distribuição de perda parcial sobre val+test incluídos: `loss_fraction_mean`/`_median` e
    contagem de casos em 4 faixas (`n_loss_zero`, `n_loss_lt5pct` `(0, 5%)`, `n_loss_5to20pct`
    `[5%, 20%)`, `n_loss_gte20pct` `[20%, 100%]`).
    """
    if not (50 < offset_percentile <= 100):
        raise ValueError(
            f"offset_percentile deve estar em (50, 100], recebido {offset_percentile}"
        )
    if (exclude_via_criterion is None) != (exclusion_pct is None):
        raise ValueError(
            "exclude_via_criterion e exclusion_pct devem ser fornecidos juntos ou nenhum dos "
            "dois."
        )

    required_columns = {
        "img_id",
        "status",
        "anchor_y",
        "anchor_x",
        "anchor_z",
        "label_y_min",
        "label_y_max",
        "label_x_min",
        "label_x_max",
        "label_z_min",
        "label_z_max",
        "volume_shape_y",
        "volume_shape_x",
        "volume_shape_z",
    }
    if exclude_via_criterion is not None:
        required_columns.add(exclude_via_criterion)
    missing = required_columns - set(records_df.columns)
    if missing:
        raise ValueError(
            f"records_df não contém as colunas esperadas: {sorted(missing)}"
        )

    ok = records_df[records_df["status"] == "ok"]
    train_df = ok[ok["img_id"].isin(train_ids)]
    val_df = ok[ok["img_id"].isin(val_ids)]
    test_df = ok[ok["img_id"].isin(test_ids)]

    if train_df.empty:
        raise ValueError("Nenhum caso de treino com status='ok' em records_df.")

    if exclude_via_criterion is not None:
        train_f, val_f, test_f, low_threshold, high_threshold = (
            _filter_splits_by_train_threshold(
                train_df, val_df, test_df, exclude_via_criterion, exclusion_pct
            )
        )
        _raise_if_any_split_empty(
            train_f, val_f, test_f, exclude_via_criterion, exclusion_pct,
            low_threshold, high_threshold,
        )
    else:
        train_f, val_f, test_f = train_df, val_df, test_df
        low_threshold = high_threshold = None

    offset_min: dict[str, float] = {}
    offset_max: dict[str, float] = {}
    for axis in ("y", "x", "z"):
        anchor_col = train_f[f"anchor_{axis}"]
        low_values = train_f[f"label_{axis}_min"] - anchor_col
        high_values = train_f[f"label_{axis}_max"] - anchor_col
        offset_min[axis] = float(np.percentile(low_values, 100.0 - offset_percentile))
        offset_max[axis] = float(np.percentile(high_values, offset_percentile))

    bbox_extent_reduced = {axis: offset_max[axis] - offset_min[axis] for axis in ("y", "x", "z")}
    bbox_extent_full_resolution = {
        axis: (offset_max[axis] - offset_min[axis]) * factor
        for axis, factor in zip(("y", "x", "z"), downscale_factors)
    }
    v_bbox_full_res = (
        bbox_extent_full_resolution["y"]
        * bbox_extent_full_resolution["x"]
        * bbox_extent_full_resolution["z"]
    )

    included_df = pd.concat([train_f, val_f, test_f], ignore_index=True)
    v_original_full_res = (
        (included_df["volume_shape_y"] * downscale_factors[0])
        * (included_df["volume_shape_x"] * downscale_factors[1])
        * (included_df["volume_shape_z"] * downscale_factors[2])
    )
    rv_per_case = 1.0 - (v_bbox_full_res / v_original_full_res)

    eval_df = pd.concat([val_f, test_f], ignore_index=True).copy()
    overlap_volume = pd.Series(1.0, index=eval_df.index)
    label_volume = pd.Series(1.0, index=eval_df.index)
    for axis in ("y", "x", "z"):
        anchor_col = eval_df[f"anchor_{axis}"]
        bbox_min = anchor_col + offset_min[axis]
        bbox_max = anchor_col + offset_max[axis]
        overlap = (
            np.minimum(eval_df[f"label_{axis}_max"], bbox_max)
            - np.maximum(eval_df[f"label_{axis}_min"], bbox_min)
        ).clip(lower=0)
        overlap_volume = overlap_volume * overlap
        label_volume = label_volume * (eval_df[f"label_{axis}_max"] - eval_df[f"label_{axis}_min"])

    loss_fraction = 1.0 - (overlap_volume / label_volume)

    return {
        "offset_percentile": float(offset_percentile),
        "exclude_via_criterion": exclude_via_criterion,
        "exclusion_pct": float(exclusion_pct) if exclusion_pct is not None else None,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "n_train_total": int(len(train_df)),
        "n_train_excluded": int(len(train_df) - len(train_f)),
        "n_train_included": int(len(train_f)),
        "n_val_total": int(len(val_df)),
        "n_val_excluded": int(len(val_df) - len(val_f)),
        "n_val_included": int(len(val_f)),
        "n_test_total": int(len(test_df)),
        "n_test_excluded": int(len(test_df) - len(test_f)),
        "n_test_included": int(len(test_f)),
        "bbox_extent_reduced": bbox_extent_reduced,
        "bbox_extent_full_resolution": bbox_extent_full_resolution,
        "rv_n_cases": int(len(included_df)),
        "rv_mean": float(rv_per_case.mean()),
        "rv_median": float(rv_per_case.median()),
        "n_cases_evaluated": int(len(eval_df)),
        "loss_fraction_mean": float(loss_fraction.mean()) if len(eval_df) else float("nan"),
        "loss_fraction_median": float(loss_fraction.median()) if len(eval_df) else float("nan"),
        "n_loss_zero": int((loss_fraction <= 1e-9).sum()),
        "n_loss_lt5pct": int(((loss_fraction > 1e-9) & (loss_fraction < 0.05)).sum()),
        "n_loss_5to20pct": int(((loss_fraction >= 0.05) & (loss_fraction < 0.20)).sum()),
        "n_loss_gte20pct": int((loss_fraction >= 0.20).sum()),
    }


__all__ = [
    "AORTA_ANCHOR_RECORD_COLUMNS",
    "segment_aorta",
    "process_case_for_aorta_anchor",
    "process_block",
    "run_aorta_anchor_batch",
    "calibrate_bbox_offsets",
    "calibrate_bbox_size_joint",
    "compute_per_case_envelope_size",
    "evaluate_exclusion_criterion",
    "evaluate_percentile_bbox_tradeoff",
    "select_cases_by_aorta_volume_percentile",
    "select_cases_by_circle_count_percentile",
    "select_cases_by_envelope_size_percentile",
    "validate_coverage",
]
