"""Diagnóstico complementar (2026-07-26, passo 2): inspeção da imagem HU bruta (sem máscara) nas
últimas fatias do FOV, para os 10 casos-problema + referência.

O passo 1 deste diagnóstico (`check_z_max_edge_prevalence.py`) mostrou que `mask_z_max ==
volume_shape_z - 1` é universal (100% numa amostra de 50 casos e nos 10 casos-problema) — não
discrimina nada por si só. Este passo verifica se há algum padrão visual/estrutural recorrente na
imagem bruta (HU) perto da última fatia do FOV que explique por que a aorta é sistematicamente
detectada até ali (estrutura anatômica real perto do desfiladeiro torácico, artefato de
reconstrução/truncamento, ou simplesmente o protocolo de aquisição cardíaca terminando ali por
design) — comparado com IMG_ID=1 (caso de referência, validado manualmente como normal).

Usa a imagem NIfTI bruta diretamente (`nib.load` + `get_fdata`), sem rodar
`load_and_preprocess_image`/`segment_aorta` — não precisa de LCC/threshold/Level-Set para só olhar
os valores HU brutos, e o eixo z não é reduzido de resolução (`DOWNSCALE_FACTORS[2] == 1`), então o
índice de fatia bate 1:1 com o espaço usado no resto do diagnóstico. Mais barato que os passos
anteriores (sem Hough nem Level-Set). NÃO reprocessa o lote completo, NÃO altera nenhum artefato
de produção.

Uso:
    uv run python src/eda/inspect_fov_top_slices_raw.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-26)", para a
interpretação.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils.nifti_io import load_raw_img_and_label

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"
IMG_DIR = DIAGNOSIS_OUTPUT_DIR / "img"

N_SLICES_FROM_TOP = 5

CASE_IDS = [622, 196, 790, 924, 777, 934, 428, 90, 23, 44, 1]  # 1 = referência


def plot_top_slices(img_id: int, block: str, raw_img: np.ndarray, n_slices: int) -> dict:
    z_max = raw_img.shape[2] - 1
    slice_indices = list(range(z_max - n_slices + 1, z_max + 1))

    # Janela HU fixa e igual entre casos (percentil da PRÓPRIA última fatia), para comparação justa.
    vmin, vmax = np.percentile(raw_img[:, :, z_max], [1, 99.5])

    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3.5))
    for ax, z in zip(axes, slice_indices):
        ax.imshow(raw_img[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"z={z}")
        ax.axis("off")
    fig.suptitle(f"img_id={img_id} (bloco {block}) — últimas {n_slices} fatias do FOV, HU bruto")
    fig.tight_layout()
    out_path = IMG_DIR / f"case_{img_id}_fov_top_raw.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    last_slice = raw_img[:, :, z_max]
    return {
        "img_id": img_id,
        "block": block,
        "z_max": z_max,
        "hu_mean_last_slice": float(last_slice.mean()),
        "hu_std_last_slice": float(last_slice.std()),
        "hu_p99_last_slice": float(np.percentile(last_slice, 99)),
        "hu_max_last_slice": float(last_slice.max()),
        "nonzero_fraction_last_slice": float((last_slice != 0).mean()),
        "image_path": str(out_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    all_cases = pd.read_csv(PRODUCTION_DIR / "all_cases.csv")

    rows = []
    for img_id in CASE_IDS:
        block = all_cases.loc[all_cases["img_id"] == img_id, "block"].iloc[0]
        print(f"Carregando img_id={img_id} (bloco {block})...", flush=True)
        base_path = DATASET_ROOT / block
        nii_img, _ = load_raw_img_and_label(f"{base_path}/{img_id}.img.nii.gz")
        raw_img = np.array(nii_img.get_fdata(), dtype=np.float32)

        result = plot_top_slices(img_id, block, raw_img, N_SLICES_FROM_TOP)
        rows.append(result)
        print(
            f"  z_max={result['z_max']} HU medio={result['hu_mean_last_slice']:.1f} "
            f"HU p99={result['hu_p99_last_slice']:.1f} HU max={result['hu_max_last_slice']:.1f} "
            f"fracao nao-zero={result['nonzero_fraction_last_slice']:.2f} -> {result['image_path']}"
        )

    summary_df = pd.DataFrame(rows)
    summary_path = DIAGNOSIS_OUTPUT_DIR / "fov_top_slices_raw_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSalvo: {summary_path}")
    print(f"Imagens salvas em: {IMG_DIR}")


if __name__ == "__main__":
    main()
