"""Diagnóstico complementar (2026-07-24, passo 2): inspeção visual dos 5 piores casos de excesso
em z (`worst_z_excess_cases.csv`, gerado por `analyze_num_circles_quintile_coverage.py`).

Hipótese a checar por inspeção visual: os casos de quintil 5 (muitos círculos: 622, 196 — 275
detectados, o teto de toda a população) mostram over-tracking (máscara da aorta se estende além
da aorta torácica, capturando outra estrutura/ruído)? Os de quintil 1 (poucos círculos: 924, 777)
mostram under-tracking (rastreamento parou cedo)? 790 (226 círculos, ainda no quintil 5 mas mais
perto do meio) é usado para checar qual dos dois padrões se aplica.

Reprocessa SÓ estes 5 casos (`load_and_preprocess_image` + `segment_aorta`, os únicos passos caros
— label e demais campos já existem em `all_cases.csv`/`coverage_*` e não são recomputados). NÃO
altera nenhum artefato de produção; salva imagens e resumo em
output/bounding_box/aorta_anchor_z_diagnosis/.

Uso:
    uv run python src/eda/inspect_worst_z_excess_cases.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-24), passo 2", para a
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

from utils.project.config import load_config_json
from utils.segmentation.aorta_anchor import segment_aorta
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DATASET_ROOT = REPO_ROOT / "dataset"
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"
WORST_Z_PATH = DIAGNOSIS_OUTPUT_DIR / "worst_z_excess_cases.csv"
IMG_DIR = DIAGNOSIS_OUTPUT_DIR / "img"

EXPECTED_HYPOTHESIS = {
    622: "over-tracking esperado (quintil 5, 275 circulos = teto da populacao)",
    196: "over-tracking esperado (quintil 5, 275 circulos = teto da populacao)",
    790: "checar qual padrao se aplica (226 circulos, quintil 5 mas mais perto do meio)",
    924: "under-tracking esperado (quintil 1, 2 circulos = piso da populacao)",
    777: "under-tracking esperado (quintil 1, 17 circulos)",
}


def plot_case(img_id: int, block: str, lcc_image: np.ndarray, aorta_mask: np.ndarray,
              label_artery: np.ndarray, bbox_z: tuple[float, float]) -> dict:
    mask_coords = np.argwhere(aorta_mask)
    mask_z_min, mask_z_max = float(mask_coords[:, 2].min()), float(mask_coords[:, 2].max())
    centroid_y = int(round(mask_coords[:, 0].mean()))
    centroid_x = int(round(mask_coords[:, 1].mean()))

    label_coords = np.argwhere(label_artery)
    label_z_min, label_z_max = float(label_coords[:, 2].min()), float(label_coords[:, 2].max())

    vmin, vmax = np.percentile(lcc_image, [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    views = [
        ("sagital (x fixo em %d)" % centroid_x, lcc_image[:, centroid_x, :], aorta_mask[:, centroid_x, :], label_artery[:, centroid_x, :]),
        ("coronal (y fixo em %d)" % centroid_y, lcc_image[centroid_y, :, :], aorta_mask[centroid_y, :, :], label_artery[centroid_y, :, :]),
    ]
    for ax, (title, img_slice, mask_slice, label_slice) in zip(axes, views):
        ax.imshow(img_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        mask_overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        ax.imshow(mask_overlay, cmap="autumn", alpha=0.5, origin="lower", aspect="auto")
        label_overlay = np.ma.masked_where(label_slice.T == 0, label_slice.T)
        ax.imshow(label_overlay, cmap="cool", alpha=0.6, origin="lower", aspect="auto")
        ax.axhline(label_z_min, color="cyan", linestyle="--", linewidth=1, label="label z_min/max")
        ax.axhline(label_z_max, color="cyan", linestyle="--", linewidth=1)
        ax.axhline(bbox_z[0], color="lime", linestyle=":", linewidth=1, label="bbox calibrado z_min/max")
        ax.axhline(bbox_z[1], color="lime", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("eixo in-plane")
        ax.set_ylabel("z (eixo axial, cresce p/ superior)")
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"img_id={img_id} (bloco {block}) — máscara da aorta (vermelho/amarelo) em z=[{mask_z_min:.0f}, "
        f"{mask_z_max:.0f}], label coronário (azul/magenta) em z=[{label_z_min:.0f}, {label_z_max:.0f}]"
    )
    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMG_DIR / f"case_{img_id}_z_inspection.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "img_id": img_id,
        "block": block,
        "mask_z_min": mask_z_min,
        "mask_z_max": mask_z_max,
        "mask_z_extent": mask_z_max - mask_z_min,
        "label_z_min": label_z_min,
        "label_z_max": label_z_max,
        "image_path": str(out_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    config = load_config_json(str(CONFIG_PATH), {})
    worst = pd.read_csv(WORST_Z_PATH)
    all_cases = pd.read_csv(PRODUCTION_DIR / "all_cases.csv")
    calibration = json.loads(
        (PRODUCTION_DIR / "calibration_imagecas_splits_train90.json").read_text(encoding="utf-8")
    )
    offset_min_z = calibration["offset_min_reduced"]["z"]
    offset_max_z = calibration["offset_max_reduced"]["z"]

    summary_rows = []
    for row in worst.itertuples():
        img_id = int(row.img_id)
        block = all_cases.loc[all_cases["img_id"] == img_id, "block"].iloc[0]
        base_path = DATASET_ROOT / block
        print(f"Processando img_id={img_id} (bloco {block})...", flush=True)

        image_data = load_and_preprocess_image(img_id, str(base_path), config)
        lcc_image = image_data["lcc_image"]
        scaled_spacing = image_data["scaled_spacing"]
        label_artery = image_data["label"] == 1

        aorta_mask, num_circles = segment_aorta(lcc_image, scaled_spacing, config)
        expected = int(row.num_circles_detected)
        if num_circles != expected:
            print(
                f"  AVISO: num_circles_detected na reexecução ({num_circles}) difere do salvo em "
                f"all_cases.csv ({expected}) — pipeline não é 100% determinístico ou config mudou."
            )

        case_val = all_cases.loc[all_cases["img_id"] == img_id].iloc[0]
        # bbox calibrado em z: reconstruído a partir de anchor_z + offsets de produção já salvos.
        bbox_z = (case_val["anchor_z"] + offset_min_z, case_val["anchor_z"] + offset_max_z)

        result = plot_case(img_id, block, lcc_image, aorta_mask, label_artery, bbox_z)
        result["num_circles_detected"] = expected
        result["hipotese_esperada"] = EXPECTED_HYPOTHESIS[img_id]
        summary_rows.append(result)
        print(
            f"  mask z=[{result['mask_z_min']:.0f}, {result['mask_z_max']:.0f}] "
            f"(extensão {result['mask_z_extent']:.0f}), label z=[{result['label_z_min']:.0f}, "
            f"{result['label_z_max']:.0f}] -> {result['image_path']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DIAGNOSIS_OUTPUT_DIR / "worst_z_excess_visual_inspection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSalvo: {summary_path}")
    print(f"Imagens salvas em: {IMG_DIR}")


if __name__ == "__main__":
    main()
