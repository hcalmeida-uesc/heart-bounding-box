"""Diagnóstico complementar (2026-07-27): objetivar a observação qualitativa de 2026-07-26
(múltiplas estruturas redondas de calibre parecido perto do topo do FOV, em casos-problema vs.
referência) com uma métrica quantificável, em amostra maior que os 11 casos inspecionados
visualmente.

NÃO roda `segment_aorta` (sem Level-Set) nem o pipeline completo — só a etapa de detecção de
círculos (`detect_aorta_circles`/`_detect_circles_in_slice` de `aorta_localization.py`), aplicada
nas 15 fatias mais próximas do topo do FOV de cada caso (mesmo ponto de partida da produção:
`detect_aorta_circles` sempre começa em `num_slices - 1`, ver código-fonte). Muito mais barato que
os diagnósticos anteriores (~5-10s/caso vs. 40-220s/caso com Level-Set) — precisa só de
`load_and_preprocess_image` (downscale + threshold + LCC, sem Hough/Level-Set) + Hough isolado.

Métrica ("candidato ambíguo"): para cada fatia analisada, o `detect_aorta_circles` real (rodado
isoladamente na sub-janela das 15 fatias do topo, com a MESMA config de produção) dá "o círculo
principal" rastreado naquela fatia. Separadamente, `_detect_circles_in_slice` (sem restrição de
ROI, `total_num_peaks_initial` picos) dá TODOS os candidatos Hough da fatia (e das fatias
adjacentes). Um candidato conta como "ambíguo" se: (a) não é o próprio círculo principal (evita
contar auto-match); (b) raio dentro de ±20% do raio do círculo principal; (c) centro a uma
distância euclidiana menor que 2x o raio médio dos dois círculos. Reporta o total de candidatos
ambíguos nas fatias analisadas e a primeira fatia (relativa ao topo, 0 = última fatia do FOV) em
que aparece, por caso.

Amostra: os 10 casos-problema já conhecidos (622, 196, 790, 924, 777, 934, 428, 90, 23, 44) + 30
casos de comparação com `aorta_voxels` entre 0,7x-1,3x da referência (336651 voxels, IMG_ID=1),
amostra estratificada por bloco do dataset (6 por bloco x 5 blocos), já disponíveis em
`all_cases.csv` sem reprocessamento algum para a seleção da amostra.

NÃO altera nenhum artefato de produção. NÃO propõe correção — só reporta se a métrica objetiva
confirma ou não a diferença vista visualmente em 2026-07-26.

Uso:
    uv run python src/eda/measure_ambiguous_circle_candidates.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-27)", para a
interpretação.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from utils.project.config import load_config_json
from utils.segmentation.aorta_localization import _detect_circles_in_slice, detect_aorta_circles
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DATASET_ROOT = REPO_ROOT / "dataset"
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"

REFERENCE_VOLUME_VOXELS = 336_651
NORMAL_VOLUME_RATIO_RANGE = (0.7, 1.3)
N_COMPARISON_PER_BLOCK = 6
SEED = 42

N_TOP_SLICES = 15
RADIUS_TOLERANCE_RATIO = 0.20
DISTANCE_THRESHOLD_RADIUS_MULTIPLE = 2.0

PROBLEM_CASES = [622, 196, 790, 924, 777, 934, 428, 90, 23, 44]


def select_comparison_sample(all_cases: pd.DataFrame) -> pd.DataFrame:
    ok = all_cases[all_cases["status"] == "ok"]
    lo = NORMAL_VOLUME_RATIO_RANGE[0] * REFERENCE_VOLUME_VOXELS
    hi = NORMAL_VOLUME_RATIO_RANGE[1] * REFERENCE_VOLUME_VOXELS
    pool = ok[
        (ok["aorta_voxels"] >= lo)
        & (ok["aorta_voxels"] <= hi)
        & (~ok["img_id"].isin(PROBLEM_CASES))
    ]
    rng = np.random.default_rng(SEED)
    samples = []
    for block, group in pool.groupby("block"):
        n = min(N_COMPARISON_PER_BLOCK, len(group))
        block_seed = int(rng.integers(0, 2**32 - 1))
        samples.append(group.sample(n=n, random_state=block_seed))
    return pd.concat(samples, ignore_index=True).sort_values(["block", "img_id"])


def analyze_case(img_id: int, block: str, config: dict) -> dict:
    circle_config = config["CIRCLE_DETECTION"]
    hough_radii = np.arange(
        circle_config["radii_start_px"],
        circle_config["radii_end_px"],
        circle_config.get("radius_step_px", 1),
    )

    base_path = DATASET_ROOT / block
    image_data = load_and_preprocess_image(img_id, str(base_path), config)
    lcc_image = image_data["lcc_image"]
    scaled_spacing = image_data["scaled_spacing"]
    pixel_spacing = (scaled_spacing[0] + scaled_spacing[1]) / 2.0

    num_slices = lcc_image.shape[2]
    n = min(N_TOP_SLICES, num_slices)
    sub_volume = lcc_image[:, :, num_slices - n : num_slices]

    tracked = detect_aorta_circles(
        sub_volume,
        hough_radii,
        pixel_spacing,
        tol_radius_mm=circle_config["tol_radius_mm"],
        tol_distance_mm=circle_config["tol_distance_mm"],
        max_slice_miss_threshold=circle_config["max_slice_miss_threshold"],
        neighbor_distance_threshold=circle_config["neighbor_distance_threshold"],
        quadrant_offset=tuple(circle_config["quadrant_offset"]),
        total_num_peaks_initial=circle_config["total_num_peaks_initial"],
        total_num_peaks=circle_config["total_num_peaks"],
        canny_sigma=circle_config["canny_sigma"],
        use_local_roi=circle_config.get("use_local_roi", True),
        local_roi_padding=circle_config.get("local_roi_padding", 20),
        out_of_tolerance_as_miss=circle_config.get("out_of_tolerance_as_miss", False),
        interpolate_missed_circles=circle_config.get("interpolate_missed_circles", True),
        candidate_selection_strategy=circle_config.get("candidate_selection_strategy", "closest"),
        use_gpu=False,
        verbose=False,
    )

    candidate_cache: dict[int, list[tuple[float, float, float]]] = {}

    def get_candidates(local_slice_idx: int) -> list[tuple[float, float, float]]:
        if local_slice_idx not in candidate_cache:
            _, cx, cy, radii = _detect_circles_in_slice(
                sub_volume[:, :, local_slice_idx],
                hough_radii,
                circle_config["total_num_peaks_initial"],
                circle_config["canny_sigma"],
                use_gpu=False,
            )
            candidate_cache[local_slice_idx] = list(zip(cx, cy, radii))
        return candidate_cache[local_slice_idx]

    ambiguous_events = []
    n_tracked_real = 0
    for circle in tracked:
        if circle.get("interpolated", False):
            continue
        n_tracked_real += 1
        local_idx = circle["slice_index"]
        main_x, main_y, main_r = circle["center_x"], circle["center_y"], circle["radius"]

        pool = []
        for neighbor_idx in (local_idx - 1, local_idx, local_idx + 1):
            if 0 <= neighbor_idx < n:
                pool.extend(get_candidates(neighbor_idx))

        for cx, cy, r in pool:
            dist = float(np.hypot(cx - main_x, cy - main_y))
            if dist < 1e-6 and abs(r - main_r) < 1e-6:
                continue  # auto-match (o proprio circulo principal reaparecendo na busca)
            mean_radius = (main_r + r) / 2.0
            if (
                mean_radius > 0
                and abs(r - main_r) / main_r <= RADIUS_TOLERANCE_RATIO
                and dist < DISTANCE_THRESHOLD_RADIUS_MULTIPLE * mean_radius
            ):
                ambiguous_events.append({"local_slice_idx": local_idx, "slice_from_top": (n - 1) - local_idx})

    n_ambiguous = len(ambiguous_events)
    first_ambiguous_slice_from_top = (
        min(e["slice_from_top"] for e in ambiguous_events) if ambiguous_events else None
    )

    return {
        "img_id": img_id,
        "block": block,
        "n_slices_analyzed": n,
        "n_tracked_real_circles": n_tracked_real,
        "n_ambiguous_candidates": n_ambiguous,
        "first_ambiguous_slice_from_top": first_ambiguous_slice_from_top,
    }


def main() -> None:
    config = load_config_json(str(CONFIG_PATH), {})
    all_cases = pd.read_csv(PRODUCTION_DIR / "all_cases.csv")

    comparison_sample = select_comparison_sample(all_cases)
    print(f"Amostra de comparação (normal): {len(comparison_sample)} casos")

    rows = []
    for img_id in PROBLEM_CASES:
        block = all_cases.loc[all_cases["img_id"] == img_id, "block"].iloc[0]
        print(f"[problema] img_id={img_id} (bloco {block})...", flush=True)
        result = analyze_case(img_id, block, config)
        result["grupo"] = "problema"
        rows.append(result)
        print(f"  ambiguos={result['n_ambiguous_candidates']} primeira_fatia={result['first_ambiguous_slice_from_top']}")

    for row in comparison_sample.itertuples():
        print(f"[normal] img_id={row.img_id} (bloco {row.block})...", flush=True)
        result = analyze_case(int(row.img_id), row.block, config)
        result["grupo"] = "normal"
        rows.append(result)
        print(f"  ambiguos={result['n_ambiguous_candidates']} primeira_fatia={result['first_ambiguous_slice_from_top']}")

    result_df = pd.DataFrame(rows)
    result_path = DIAGNOSIS_OUTPUT_DIR / "ambiguous_circle_candidates_summary.csv"
    result_df.to_csv(result_path, index=False)
    print(f"\nSalvo: {result_path}")

    print("\n=== Resumo por grupo ===")
    summary = result_df.groupby("grupo")["n_ambiguous_candidates"].agg(["count", "mean", "median", "std", "min", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
