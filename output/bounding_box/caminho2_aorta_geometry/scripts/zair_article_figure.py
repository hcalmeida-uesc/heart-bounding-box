"""Figura de controle da ROI de Zair fiel ao artigo (estilo da Fig. 4a), 8 casos fora do teste.

Linha 1: 4 casos típicos (R perto da mediana). Linha 2: os 4 casos de maior perda com o círculo
recalibrado. Fundo: fatia mediana do label (o label aparece em contorno só para mostrar a perda)."""
import json
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from c2_extract import ORIG, block_of  # noqa: E402

HERE = Path(__file__).parent
REPO = Path("D:/Doutorado/heart-bounding-box")
OUT = REPO / "output/bounding_box/caminho2_aorta_geometry/zair_article"
cs = pd.read_csv(OUT / "zair_article_cases.csv")
zs = pd.read_csv(HERE / "zair_slice.csv").set_index("img_id")
k = json.loads((OUT / "zair_article_calibration.json").read_text())["k_circulo"]
per = pd.read_csv(OUT / "zair_article_axial_percase.csv", index_col=0)
col = [c for c in per.columns if c.startswith("792: todos") and "Círculo recalibrado" in c and "produção" not in c][0]
tv = cs[(cs.split == "trainval") & cs.zair_ok]
typ = tv.assign(dd=(tv.R2_mm - tv.R2_mm.median()).abs()).nsmallest(4, "dd")
worst = tv.set_index("img_id").loc[per[col].dropna().nlargest(4).index].rename_axis("img_id").reset_index()
sel = [("típico", r) for _, r in typ.iterrows()] + [("maior perda", r) for _, r in worst.iterrows()]

fig, axs = plt.subplots(2, 4, figsize=(20, 10.8))
for ax, (kind, r) in zip(axs.flat, sel):
    i = int(r.img_id)
    blk = ORIG / block_of(i)
    nii = nib.load(str(blk / f"{i}.img.nii.gz"))
    sp = float(np.mean(nii.header.get_zooms()[:2]))
    z = int(r.z_sel)
    ax.imshow(np.asarray(nii.dataobj[:, :, z], dtype=np.float32), cmap="gray", vmin=-200, vmax=700)
    lab = np.asarray(nib.load(str(blk / f"{i}.label.nii.gz")).dataobj) == 1
    proj = lab.any(axis=2)
    ax.contour(proj.astype(float), levels=[0.5], colors="orange", linewidths=1.0)
    q = zs.loc[i]
    for c0, c1, rr, name, c in ((q.asc_c0_red, q.asc_c1_red, q.asc_r_red, "A", "red"),
                                (q.desc_c0_red, q.desc_c1_red, q.desc_r_red, "B", "blue")):
        ax.add_patch(plt.Circle((2 * c1, 2 * c0), 2 * rr, fill=False, color=c, lw=1.5))
        ax.text(2 * c1 + 2 * rr, 2 * c0, name, color=c, fontsize=12, weight="bold")
    ax.plot(r.Cc1, r.Cc0, "x", color="yellow", ms=12, mew=2)
    for rad, c, ls in ((r.R2_mm / sp, "cyan", "--"), (k * r.R2_mm / sp, "lime", "-")):
        ax.add_patch(plt.Circle((r.Cc1, r.Cc0), rad, fill=False, color=c, lw=1.8, ls=ls))
    loss = per.loc[i, col]
    ax.set_title(f"[{kind}] caso {i}, fatia z = {z}\nR = {r.R2_mm:.0f} mm (D = {r.D_mm:.0f}, |ΔAP| = {r.dAP_mm:.0f})"
                 f"; perda com k = {k:.2f}: {loss:.1f}%", fontsize=11)
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_xlabel("eixo 1 (ântero-posterior) → anterior")
    ax.set_ylabel("eixo 0 (látero-lateral) → esquerda")
fig.suptitle("Tracejado ciano: R do artigo (k = 1). Verde: R recalibrado. X amarelo: centro C. "
             "Laranja: projeção axial do label coronário.", fontsize=12)
fig.tight_layout()
out = REPO / "docs/img/caminho2_zair_artigo_controle.png"
fig.savefig(out, dpi=65)
print(out, [int(r.img_id) for _, r in sel])
