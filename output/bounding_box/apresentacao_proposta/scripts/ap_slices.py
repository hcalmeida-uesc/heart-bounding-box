"""Apresentação da proposta: cortes axiais de um caso por tipo de detecção (caixa fora da amostra).

Lê casos_cortes.csv (ap_freeze.py), a imagem e o label originais (só leitura) e os círculos detectados
refeitos por c2_extract.py para os casos escolhidos (c2_cache/ local; conferidos contra c2_cases.csv).
Eixos LAS: 0 → esquerda do paciente, 1 → anterior, 2 → superior."""
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from c2_extract import ORIG, block_of  # noqa: E402
from c2_select import select_desc  # noqa: E402

REPO = Path("D:/Doutorado/heart-bounding-box")
OUT = REPO / "output/bounding_box/apresentacao_proposta"
IMG = REPO / "docs/img/apresentacao_proposta"
IMG.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(HERE / "c2_extract_meta.csv").set_index("img_id")
sel = pd.read_csv(OUT / "casos_cortes.csv")
LEVEL, WIDTH = 150, 600
C_BOX, C_COR, C_ASC, C_DESC = "#ffd400", "#ff2d2d", "#00d5ff", "#5dff3a"
plt.rcParams.update({"font.size": 16, "svg.fonttype": "none"})


def mm_label(v):
    v = int(round(v))
    return "0 mm" if v == 0 else (f"+{v} mm" if v > 0 else f"\u2212{-v} mm")


rows = []
for _, r in sel.iterrows():
    i, t = int(r.img_id), int(r.tipo)
    nii = nib.load(str(ORIG / block_of(i) / f"{i}.img.nii.gz"))
    assert nib.aff2axcodes(nii.affine) == ("L", "A", "S")
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    spz = float(nii.header.get_zooms()[2])
    lab = np.asarray(nib.load(str(ORIG / block_of(i) / f"{i}.label.nii.gz")).dataobj) == 1
    lo = np.array([r.caixa_lo0, r.caixa_lo1, r.caixa_lo2], int)
    hi = np.array([r.caixa_hi0, r.caixa_hi1, r.caixa_hi2], int)
    c0, c1 = (lo[0] + hi[0]) / 2 - 0.5, (lo[1] + hi[1]) / 2 - 0.5        # centro em coordenadas de voxel
    zc = int(round((lo[2] + hi[2] - 1) / 2))
    s = int(min((hi[2] - 1 - zc) // 3, (zc - lo[2]) // 2))
    while s > 1 and abs(s * spz - round(s * spz)) > 1e-6:                # passo inteiro em mm, se possível
        s -= 1
    zs = [zc + k * s for k in (3, 2, 1, 0, -1, -2)]                    # cranial → caudal
    assert all(lo[2] <= z < hi[2] for z in zs)
    asc = desc = np.zeros((0, 9))
    if t in (1, 2, 3):
        d = np.load(HERE / f"c2_cache/{i}.npz")
        asc = d["asc"]
        if t in (1, 2):
            desc, _, _ = select_desc(asc, d["desc"], float(meta.loc[i, "rsp0"]))
    dashed = t in (2, 3)

    fig, axs = plt.subplots(2, 3, figsize=(15, 12.8))
    for ax, z in zip(axs.flat, zs):
        sl = vol[:, :, z]
        # convenção radiológica: linhas = eixo 1 (anterior em cima), colunas = eixo 0 (esquerda à direita)
        ax.imshow(sl.T, origin="lower", cmap="gray", vmin=LEVEL - WIDTH / 2, vmax=LEVEL + WIDTH / 2,
                  interpolation="nearest")
        cm = np.ma.masked_where(~lab[:, :, z].T, np.ones_like(sl.T))
        ax.imshow(cm, origin="lower", cmap=matplotlib.colors.ListedColormap([C_COR]), vmin=0, vmax=1,
                  interpolation="nearest")
        ax.add_patch(Rectangle((lo[0] - 0.5, lo[1] - 0.5), hi[0] - lo[0], hi[1] - lo[1], fill=False,
                               ec=C_BOX, lw=2.2))
        if z == zc:
            ax.plot(c0, c1, "+", color=C_BOX, ms=26, mew=3)
        for arr, col in ((asc, C_ASC), (desc, C_DESC)):
            for q in arr[arr[:, 0] == z]:
                ax.add_patch(plt.Circle((2 * q[1], 2 * q[2]), 2 * q[3], fill=False,
                                        ec=(C_ASC if dashed else col), lw=2.2, ls=("--" if dashed else "-")))
        ax.set_xlim(-0.5, sl.shape[0] - 0.5)
        ax.set_ylim(-0.5, sl.shape[1] - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(mm_label((z - zc) * spz), fontsize=18)
        rows.append(dict(tipo=t, img_id=i, z=z, z_centro=zc, posicao_mm=(z - zc) * spz,
                         voxels_coronaria=int(lab[:, :, z].sum()), circ_asc=int((asc[:, 0] == z).sum()),
                         circ_desc=int((desc[:, 0] == z).sum())))
    h = [Line2D([], [], color=C_BOX, lw=2.2, label="Caixa"),
         Line2D([], [], color=C_BOX, marker="+", ms=18, mew=3, ls="none", label="Centro da caixa"),
         Patch(color=C_COR, label="Coronárias (anotação)")]
    if t == 1:
        h += [Line2D([], [], color=C_ASC, lw=2.2, label="Aorta ascendente"),
              Line2D([], [], color=C_DESC, lw=2.2, label="Aorta descendente")]
    elif dashed and any(((a[:, 0] == z).any() for a in (asc, desc) for z in zs)):   # só se aparece nos cortes
        h += [Line2D([], [], color=C_ASC, lw=2.2, ls="--", label="Aorta detectada (descartada)")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=16)
    fig.suptitle(r.titulo, fontsize=21, y=0.975)
    fig.text(0.5, 0.925, f"Perda das coronárias neste caso: {r.perda_caso:.1f}%".replace(".", ","),
             ha="center", fontsize=17)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.865, bottom=0.07, wspace=0.03, hspace=0.12)
    for ext in ("png", "svg"):
        fig.savefig(IMG / f"cortes_tipo{t}.{ext}", dpi=300)
    plt.close(fig)
    print("ok", t, i, zs)
pd.DataFrame(rows).to_csv(OUT / "cortes_posicoes.csv", index=False)
