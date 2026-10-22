"""Caminho 2, Passo 1: características por caso a partir de c2_cache/ (sem rótulo).

Saída: c2_features.csv. Coordenadas em voxels de resolução integral (reduzida × (2, 2, 1), mesma
convenção da âncora de produção) e em mm (× spacing do cabeçalho NIfTI)."""
from pathlib import Path

import numpy as np
import pandas as pd

from c2_select import MIN_SLICES, select_desc

HERE = Path(__file__).parent
ROOT_WIN_MM = 20.0   # janela da ascendente: da raiz (menor z rastreado) até 20 mm acima
F = np.array([2.0, 2.0, 1.0])


def case_features(i, meta):
    d = np.load(HERE / f"c2_cache/{i}.npz")
    asc, desc = d["asc"], d["desc"]
    rsp = float(meta.rsp0)
    fsp = np.array([meta.fsp0, meta.fsp1, meta.fsp2])
    out = dict(img_id=i, n_asc=len(asc))
    if len(asc) == 0:
        out["asc_found"] = False
        return out
    real = asc[asc[:, 5] == 0]
    z_root, z_top = asc[:, 0].min(), asc[:, 0].max()
    win = real[real[:, 0] <= z_root + ROOT_WIN_MM / fsp[2]]
    out.update(asc_found=True, n_asc_real=len(real), z_root=z_root, z_top=z_top, n_root_win=len(win))
    asc_c = np.array([np.median(win[:, 1]), np.median(win[:, 2]), np.median(win[:, 0])]) * F
    asc_r = np.median(win[:, 3]) * 2.0  # voxels integrais (eixo 0/1)
    for j, a in enumerate(("0", "1", "2")):
        out[f"asc_c{a}_vox"] = asc_c[j]
        out[f"asc_c{a}_mm"] = asc_c[j] * fsp[j]
    out["asc_D_vox"] = 2 * asc_r
    out["asc_D_mm"] = 2 * asc_r * (fsp[0] + fsp[1]) / 2
    out["asc_dHU_root_win"] = float(np.median(win[:, 6]))
    ch, mode, sup = select_desc(asc, desc, rsp)
    out.update(n_desc_slices=len(ch), desc_support=sup, desc_frac_slices=len(ch) / max(len(real), 1))
    if len(ch) < MIN_SLICES:
        out["desc_found"] = False
        return out
    dc = np.array([np.median(ch[:, 1]), np.median(ch[:, 2]), np.median(ch[:, 0])]) * F
    dr = np.median(ch[:, 3]) * 2.0
    out["desc_found"] = True
    for j, a in enumerate(("0", "1", "2")):
        out[f"desc_c{a}_vox"] = dc[j]
        out[f"desc_c{a}_mm"] = dc[j] * fsp[j]
    out["desc_D_vox"] = 2 * dr
    out["desc_D_mm"] = 2 * dr * (fsp[0] + fsp[1]) / 2
    # dispersão do centro escolhido entre fatias (mm) e diferença de HU para a ascendente
    out["desc_c_iqr_mm"] = float(np.hypot(*(np.subtract(*np.percentile(ch[:, 1:3], [75, 25], axis=0))
                                            * 2 * fsp[:2])))
    am = {int(r[0]): r for r in asc}
    out["desc_dHU_med"] = float(np.median([r[5] - am[int(r[0])][6] for r in ch]))
    # distância de borda do FOV ao círculo (mm, plano axial)
    edge = min(dc[0], dc[1], 2 * meta.shape0 - dc[0], 2 * meta.shape1 - dc[1]) - dr
    out["desc_edge_mm"] = edge * fsp[0]
    # distância entre centros no plano axial
    for unit, sp in (("vox", np.ones(3)), ("mm", fsp)):
        d0 = (dc[0] - asc_c[0]) * sp[0]
        d1 = (dc[1] - asc_c[1]) * sp[1]
        out[f"dLL_{unit}"] = d0
        out[f"dAP_{unit}"] = d1
        out[f"d_{unit}"] = float(np.hypot(d0, d1))
        out[f"R_{unit}"] = out[f"d_{unit}"] + 2 * out[f"asc_D_{unit}"]
    return out


if __name__ == "__main__":
    meta = pd.read_csv(HERE / "c2_extract_meta.csv").set_index("img_id")
    meta = meta[meta.index.map(lambda i: (HERE / f"c2_cache/{i}.npz").exists())]
    rows = [case_features(i, m) for i, m in meta.iterrows()]
    df = pd.DataFrame(rows)
    df = df.merge(meta.reset_index()[["img_id", "shape0", "shape1", "shape2", "fsp0", "fsp1", "fsp2"]], on="img_id")
    df.to_csv(HERE / "c2_features.csv", index=False)
    print(len(df), df.asc_found.sum(), df.desc_found.sum())
    print(df.describe().T.round(2).to_string())
