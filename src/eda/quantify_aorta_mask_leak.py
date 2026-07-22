"""Diagnóstico complementar (2026-07-25): quantificação do "leak" (over-tracking) vs. fragmento
truncado (under-tracking) na máscara da aorta, para dois grupos de 5 casos:

1. Os 5 piores casos de excesso em z já inspecionados visualmente
   (`inspect_worst_z_excess_cases.py`, 622/196/790 = over-tracking, 924/777 = under-tracking).
2. Os 5 casos que definiram os extremos por eixo na calibração original (2026-07-16):
   934 (y_min), 428 (y_max/x_min), 90 (x_max), 23 (z_min), 44 (z_max) — os que produziram
   `bbox_extent_reduced` = 240×207×313 (achado bloqueante, `Rv` ~5%).

Para cada caso: volume total da máscara (voxels), comparado à referência de IMG_ID=1 (336651
voxels, validado manualmente como caso "normal" — ver seção "Execução do lote completo", linha
~116), e o perfil de área de seção transversal por fatia axial ao longo do range z da máscara,
usado para localizar saltos abruptos de área (assinatura de leak) vs. fragmento pequeno e
relativamente constante (assinatura de truncamento).

NÃO reprocessa o lote inteiro: só estes 10 casos precisam de `load_and_preprocess_image` +
`segment_aorta` (mesmo custo dos 5 já rodados no diagnóstico anterior) — o volume total já estava
em `all_cases.csv` (`aorta_voxels`) para todos os 10 sem reprocessar nada, mas o perfil de área por
fatia não é um campo salvo em produção, por isso o reprocessamento é necessário para essa parte.
NÃO altera nenhum artefato de produção.

Uso:
    uv run python src/eda/quantify_aorta_mask_leak.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-25)", para a
interpretação.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.inspect_worst_z_excess_cases import plot_case
from utils.project.config import load_config_json
from utils.segmentation.aorta_anchor import segment_aorta
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DATASET_ROOT = REPO_ROOT / "dataset"
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"
IMG_DIR = DIAGNOSIS_OUTPUT_DIR / "img"

REFERENCE_VOLUME_VOXELS = 336_651  # IMG_ID=1, validado manualmente como caso "normal" (docs)

WORST_COVERAGE_CASES = [
    {"img_id": 622, "grupo": "piores_cobertura_z", "eixo_calibracao": ""},
    {"img_id": 196, "grupo": "piores_cobertura_z", "eixo_calibracao": ""},
    {"img_id": 790, "grupo": "piores_cobertura_z", "eixo_calibracao": ""},
    {"img_id": 924, "grupo": "piores_cobertura_z", "eixo_calibracao": ""},
    {"img_id": 777, "grupo": "piores_cobertura_z", "eixo_calibracao": ""},
]
CALIBRATION_EXTREME_CASES = [
    {"img_id": 934, "grupo": "extremo_calibracao", "eixo_calibracao": "y_min"},
    {"img_id": 428, "grupo": "extremo_calibracao", "eixo_calibracao": "y_max/x_min"},
    {"img_id": 90, "grupo": "extremo_calibracao", "eixo_calibracao": "x_max"},
    {"img_id": 23, "grupo": "extremo_calibracao", "eixo_calibracao": "z_min"},
    {"img_id": 44, "grupo": "extremo_calibracao", "eixo_calibracao": "z_max"},
]
ALL_CASES = WORST_COVERAGE_CASES + CALIBRATION_EXTREME_CASES


def plot_area_profile(img_id: int, area_per_slice: np.ndarray, z_min: int, z_max: int, jump_z: int | None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    z_range = np.arange(z_min, z_max + 1)
    ax.plot(z_range, area_per_slice[z_min : z_max + 1], marker=".", markersize=3, linewidth=1)
    if jump_z is not None:
        ax.axvline(jump_z, color="red", linestyle="--", linewidth=1, label=f"maior salto (z={jump_z})")
        ax.legend()
    ax.set_xlabel("z (fatia axial)")
    ax.set_ylabel("área da máscara na fatia (voxels)")
    ax.set_title(f"img_id={img_id} — área de seção transversal da máscara da aorta por fatia")
    fig.tight_layout()
    out_path = IMG_DIR / f"case_{img_id}_area_profile.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def analyze_case(img_id: int, block: str, config: dict, calibration: dict) -> dict:
    base_path = DATASET_ROOT / block
    image_data = load_and_preprocess_image(img_id, str(base_path), config)
    lcc_image = image_data["lcc_image"]
    scaled_spacing = image_data["scaled_spacing"]
    label_artery = image_data["label"] == 1

    aorta_mask, num_circles = segment_aorta(lcc_image, scaled_spacing, config)
    volume = int(aorta_mask.sum())

    area_per_slice = aorta_mask.sum(axis=(0, 1)).astype(float)
    nonzero_z = np.flatnonzero(area_per_slice)
    z_min, z_max = int(nonzero_z.min()), int(nonzero_z.max())
    active_area = area_per_slice[z_min : z_max + 1]

    diffs = np.abs(np.diff(active_area))
    if len(diffs) > 0:
        jump_idx = int(np.argmax(diffs))
        jump_z = z_min + jump_idx + 1  # fatia onde o salto "chega"
        jump_delta = float(diffs[jump_idx])
        median_area = float(np.median(active_area[active_area > 0])) if (active_area > 0).any() else 0.0
        jump_ratio = jump_delta / median_area if median_area > 0 else float("nan")
    else:
        jump_z, jump_delta, jump_ratio = None, 0.0, float("nan")

    cv = float(active_area.std() / active_area.mean()) if active_area.mean() > 0 else float("nan")

    area_profile_path = plot_area_profile(img_id, area_per_slice, z_min, z_max, jump_z)

    all_cases = pd.read_csv(PRODUCTION_DIR / "all_cases.csv")
    case_val = all_cases.loc[all_cases["img_id"] == img_id].iloc[0]
    offset_min_z = calibration["offset_min_reduced"]["z"]
    offset_max_z = calibration["offset_max_reduced"]["z"]
    bbox_z = (case_val["anchor_z"] + offset_min_z, case_val["anchor_z"] + offset_max_z)

    visual_result = plot_case(img_id, block, lcc_image, aorta_mask, label_artery, bbox_z)

    return {
        "img_id": img_id,
        "block": block,
        "num_circles_detected": num_circles,
        "mask_volume_voxels": volume,
        "volume_ratio_vs_ref": volume / REFERENCE_VOLUME_VOXELS,
        "mask_z_min": z_min,
        "mask_z_max": z_max,
        "mask_z_extent": z_max - z_min,
        "jump_z": jump_z,
        "jump_delta_voxels": jump_delta,
        "jump_ratio_vs_median_area": jump_ratio,
        "area_coefficient_of_variation": cv,
        "area_profile_image": str(area_profile_path.relative_to(REPO_ROOT)),
        "z_inspection_image": visual_result["image_path"],
    }


def main() -> None:
    config = load_config_json(str(CONFIG_PATH), {})
    calibration = json.loads(
        (PRODUCTION_DIR / "calibration_imagecas_splits_train90.json").read_text(encoding="utf-8")
    )
    all_cases = pd.read_csv(PRODUCTION_DIR / "all_cases.csv")

    rows = []
    for case in ALL_CASES:
        img_id = case["img_id"]
        block = all_cases.loc[all_cases["img_id"] == img_id, "block"].iloc[0]
        print(f"Processando img_id={img_id} (grupo={case['grupo']}, bloco {block})...", flush=True)
        result = analyze_case(img_id, block, config, calibration)
        result["grupo"] = case["grupo"]
        result["eixo_calibracao"] = case["eixo_calibracao"]
        rows.append(result)
        print(
            f"  volume={result['mask_volume_voxels']} ({result['volume_ratio_vs_ref']:.2f}x ref) "
            f"z=[{result['mask_z_min']}, {result['mask_z_max']}] "
            f"maior_salto: z={result['jump_z']} delta={result['jump_delta_voxels']:.0f} "
            f"({result['jump_ratio_vs_median_area']:.2f}x mediana) CV={result['area_coefficient_of_variation']:.2f}"
        )

    summary_df = pd.DataFrame(rows)
    ordered_cols = [
        "grupo", "img_id", "block", "eixo_calibracao", "num_circles_detected",
        "mask_volume_voxels", "volume_ratio_vs_ref", "mask_z_min", "mask_z_max", "mask_z_extent",
        "jump_z", "jump_delta_voxels", "jump_ratio_vs_median_area", "area_coefficient_of_variation",
        "area_profile_image", "z_inspection_image",
    ]
    summary_df = summary_df[ordered_cols]
    summary_path = DIAGNOSIS_OUTPUT_DIR / "mask_leak_quantification_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSalvo: {summary_path}")
    print(f"Imagens salvas em: {IMG_DIR}")


if __name__ == "__main__":
    main()
