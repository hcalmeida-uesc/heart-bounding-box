"""Equivalência: compute_composite_roi (produção) × análise exploratória (a8_calibrate), mesmos parâmetros.

Fonte das características: "csv" (c2_cases.csv + all_cases.csv, as entradas da análise) ou "cache"
(c2_cache regenerado → summarize_aorta_geometry, + all_cases.csv). Exploratório: a8.make_box com a
calibração congelada (calib("A8c", 792, crossfit=True)) no p congelado, para os 792 e os 208."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import a7_calibrate as m
import a8_calibrate as a8

sys.path.insert(0, str(m.REPO / "src"))
from utils.segmentation.roi_composite_anchor import compute_composite_roi, load_frozen_params, summarize_aorta_geometry  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "csv"
PARAMS = load_frozen_params(m.REPO / "output/bounding_box/apresentacao_proposta/a8c_frozen.json")
P = PARAMS["p"]
cal = a8.calib("A8c", m.TV, True)
CACHE = m.REPO / "output/bounding_box/caminho2_aorta_geometry/scripts/c2_cache"


def feats_csv(j):
    def v(arr):
        return None if np.isnan(arr[j]).any() else arr[j].tolist()
    d = m.df.d_mm.iloc[j]
    return dict(aorta_anchor=v(m.A1), asc_center=v(m.ASC), desc_center=v(m.DESC), desc_found=bool(m.df.desc_found.iloc[j]),
                d_mm=None if np.isnan(d) else float(d))


def feats_cache(j):
    i = int(m.IDS[j])
    z = np.load(CACHE / f"{i}.npz")
    r = m.df.iloc[j]
    fsp = (float(r.spy), float(r.spx), float(r.spz))
    g = summarize_aorta_geometry(z["asc"], z["desc"], fsp[0] * 2.0, fsp)
    a1 = None if np.isnan(m.A1[j]).any() else m.A1[j].tolist()
    return dict(aorta_anchor=a1, **g)


rows = []
for name, idx in (("792", m.TV), ("208", m.TE)):
    lo, hi, fb = a8.make_box(cal, P, idx)
    for n, j in enumerate(idx):
        f = feats_csv(j) if SRC == "csv" else feats_cache(j)
        out = compute_composite_roi(f, tuple(int(s) for s in m.S[j]), PARAMS)
        fc = feats_csv(j)
        dfeat = 0.0
        for k in ("asc_center", "desc_center"):
            a, b = f.get(k), fc[k]
            if (a is None) != (b is None):
                dfeat = np.inf
            elif a is not None:
                dfeat = max(dfeat, float(np.abs(np.subtract(a, b)).max()))
        if (f.get("d_mm") is None) != (fc["d_mm"] is None):
            dfeat = np.inf
        elif f.get("d_mm") is not None:
            dfeat = max(dfeat, abs(f["d_mm"] - fc["d_mm"]))
        rows.append(dict(conjunto=name, img_id=int(m.IDS[j]), regime_prod=out["regime"], fb_expl=bool(fb[n]),
                         regime_igual=(out["regime"] == "aorta_nao_confiavel") == bool(fb[n]),
                         caixa_igual=(tuple(out["bbox_min"]) == tuple(lo[n].astype(int))) and (tuple(out["bbox_max"]) == tuple(hi[n].astype(int))),
                         dif_max_caixa=int(max(np.abs(np.subtract(out["bbox_min"], lo[n])).max(), np.abs(np.subtract(out["bbox_max"], hi[n])).max())),
                         dif_caracteristicas=dfeat))
r = pd.DataFrame(rows)
out = m.REPO / f"output/bounding_box/cropped_option3/equivalencia_{SRC}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
r.to_csv(out, index=False)
print(r.groupby("conjunto").agg(n=("img_id", "size"), regime_igual=("regime_igual", "sum"), caixa_igual=("caixa_igual", "sum"),
                                 dif_max_caixa=("dif_max_caixa", "max"), dif_caract=("dif_caracteristicas", "max"),
                                 fallback=("fb_expl", "sum")))
