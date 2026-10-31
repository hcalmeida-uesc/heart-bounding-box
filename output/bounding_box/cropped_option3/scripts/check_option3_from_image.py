"""Confere o caminho completo a partir da imagem (composite_roi_from_image: pré-processamento, Hough,
Level Set, descendente, caixa) contra o caminho por cache usado na geração, caso a caso, numa amostra.

Amostra (seed 42): 4 casos de cada regime nos 792, 2 de cada regime nos 208 e os 2 primeiros casos sem
aorta de produção. Compara: âncora de produção com all_cases.csv; centros e d_mm com c2_cases.csv;
regime e caixa com equivalencia_csv.csv (análise exploratória com os mesmos parâmetros). Só lê a imagem.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.roi_composite_anchor import (  # noqa: E402
    composite_roi_from_image, compute_composite_roi, load_frozen_params)

OUT = REPO / "output/bounding_box/cropped_option3"
CFG = json.loads((REPO / "config/pipeline_config.json").read_text())
PARAMS = load_frozen_params(REPO / "output/bounding_box/apresentacao_proposta/a8c_frozen.json")


def block_of(i):
    lo = (i - 1) // 200 * 200 + 1
    return f"{lo}-{lo + 199}"


def run(i):
    t0 = time.perf_counter()
    r = composite_roi_from_image(Path("D:/Doutorado/dataset/original") / block_of(i) / f"{i}.img.nii.gz", CFG, PARAMS)
    return i, r, time.perf_counter() - t0


if __name__ == "__main__":
    eq = pd.read_csv(OUT / "equivalencia_csv.csv")
    ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv").set_index("img_id")
    c2 = pd.read_csv(REPO / "output/bounding_box/caminho2_aorta_geometry/c2_cases.csv").set_index("img_id")
    rng = np.random.default_rng(42)
    ids = []
    for cj, n in (("792", 4), ("208", 2)):
        for fb in (False, True):
            pool = eq[(eq.conjunto.astype(str) == cj) & (eq.fb_expl == fb)].img_id.to_numpy()
            ids += sorted(rng.choice(pool, n, replace=False).tolist())
    ids += sorted(ac[ac.status != "ok"].index[:2].tolist())
    print("amostra:", ids, flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, r, sec in ex.map(run, ids):
            f = r["features"]
            e = eq[eq.img_id == i].iloc[0]
            q = c2.loc[i]
            a = ac.loc[i]
            a1_ref = None if a.status != "ok" else np.array([a.anchor_y, a.anchor_x, a.anchor_z]) * [2, 2, 1]
            a1 = f["aorta_anchor"]
            asc_ref = None if not bool(q.asc_found) else np.array([q.asc_c0_vox, q.asc_c1_vox, q.asc_c2_vox])
            desc_ref = None if not bool(q.desc_found) else np.array([q.desc_c0_vox, q.desc_c1_vox, q.desc_c2_vox])

            def dif(x, y):
                if (x is None) != (y is None):
                    return np.inf
                return 0.0 if x is None else float(np.abs(np.subtract(x, y)).max())
            # caixa pelo caminho das entradas da análise (igual à exploratória em 1000/1000, equivalencia_csv.csv)
            ref = compute_composite_roi(dict(aorta_anchor=None if a1_ref is None else a1_ref.tolist(),
                                             asc_center=None if asc_ref is None else asc_ref.tolist(),
                                             desc_center=None if desc_ref is None else desc_ref.tolist(),
                                             desc_found=bool(q.desc_found), d_mm=None if pd.isna(q.d_mm) else float(q.d_mm)),
                                        r["features"]["volume_shape_full"], PARAMS)
            rows.append(dict(img_id=i, conjunto=e.conjunto, segundos=sec, regime=r["regime"],
                             regime_igual=(r["regime"] == "aorta_nao_confiavel") == bool(e.fb_expl),
                             dif_aorta_producao=dif(a1, a1_ref), dif_ascendente=dif(f["asc_center"], asc_ref),
                             dif_descendente=dif(f["desc_center"], desc_ref),
                             dif_d_mm=dif(f["d_mm"], None if pd.isna(q.d_mm) else q.d_mm),
                             aorta_voxels=f["aorta_voxels"], aorta_voxels_ref=a.aorta_voxels,
                             caixa_igual=(r["bbox_min"] == ref["bbox_min"]) and (r["bbox_max"] == ref["bbox_max"]),
                             bbox_min=list(r["bbox_min"]), bbox_max=list(r["bbox_max"])))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "checagem_a_partir_da_imagem.csv", index=False)
    print(res.drop(columns=["bbox_min", "bbox_max"]).to_string(index=False))
