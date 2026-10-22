"""Caminho 2: figura de controle visual da detecção (12 casos fora do teste).

4 típicos (d perto da mediana, descendente em >= 90% das fatias), 4 com d extremo (2 menores, 2 maiores)
e 4 com falha ou detecção duvidosa. Corte axial em resolução integral, círculos convertidos × 2."""
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from c2_extract import ORIG, block_of  # noqa: E402
from c2_select import select_desc  # noqa: E402

HERE = Path(__file__).parent
REPO = Path("D:/Doutorado/heart-bounding-box")
cases = pd.read_csv(REPO / "output/bounding_box/caminho2_aorta_geometry/c2_cases.csv")
meta = pd.read_csv(HERE / "c2_extract_meta.csv").set_index("img_id")
tv = cases[cases.split == "trainval"]
ok = tv[tv.ok]
typ = ok[ok.desc_frac_slices >= 0.9].assign(dd=lambda x: (x.d_mm - ok.d_mm.median()).abs()).nsmallest(4, "dd")
ext = pd.concat([ok.nsmallest(2, "d_mm"), ok.nlargest(2, "d_mm")])
fail = tv[~tv.desc_found & tv.asc_found]
dub = ok.assign(q=lambda x: x.desc_frac_slices).nsmallest(4, "q")
bad = pd.concat([fail.head(2), dub]).head(4)
sel = [("típico", r) for _, r in typ.iterrows()] + [("d extremo", r) for _, r in ext.iterrows()] + \
      [("falha/duvidoso", r) for _, r in bad.iterrows()]

fig, axs = plt.subplots(3, 4, figsize=(20, 15.5))
for ax, (kind, r) in zip(axs.flat, sel):
    i = int(r.img_id)
    d = np.load(HERE / f"c2_cache/{i}.npz")
    asc, desc = d["asc"], d["desc"]
    ch, mode, _ = select_desc(asc, desc, float(meta.loc[i, "rsp0"]))
    if len(ch):
        z = int(ch[np.argmin(np.abs(ch[:, 0] - np.median(ch[:, 0]))), 0])
    else:
        z = int(asc[len(asc) // 3, 0])
    img = np.asarray(nib.load(str(ORIG / block_of(i) / f"{i}.img.nii.gz")).dataobj[:, :, z], dtype=np.float32)
    ax.imshow(img, cmap="gray", vmin=-200, vmax=700)
    a = asc[asc[:, 0] == z]
    if len(a):
        a = a[0]
        ax.add_patch(plt.Circle((2 * a[2], 2 * a[1]), 2 * a[3], fill=False, color="red", lw=1.8))
    c = ch[ch[:, 0] == z] if len(ch) else []
    for q in c:
        ax.add_patch(plt.Circle((2 * q[2], 2 * q[1]), 2 * q[3], fill=False, color="lime", lw=1.8))
    if r.desc_found:
        ax.plot(r.desc_c1_vox, r.desc_c0_vox, "+", color="lime", ms=14, mew=2)
        ax.plot(r.asc_c1_vox, r.asc_c0_vox, "+", color="red", ms=14, mew=2)
        ax.plot((r.asc_c1_vox + r.desc_c1_vox) / 2, (r.asc_c0_vox + r.desc_c0_vox) / 2, "x", color="yellow",
                ms=12, mew=2)
        txt = (f"d = {r.d_mm:.0f} mm; D asc = {r.asc_D_mm:.0f} mm; D desc = {r.desc_D_mm:.0f} mm\n"
               f"descendente em {int(r.n_desc_slices)}/{int(r.n_asc_real)} fatias")
    else:
        txt = f"descendente não encontrada ({int(r.n_desc_slices) if pd.notna(r.n_desc_slices) else 0} fatias)"
    ax.set_title(f"[{kind}] caso {i}, z = {z}\n{txt}", fontsize=11)
    ax.set_xlabel("eixo 1 (ântero-posterior) → anterior")
    ax.set_ylabel("eixo 0 (látero-lateral) → esquerda")
fig.tight_layout()
out = REPO / "docs/img/caminho2_controle_deteccao.png"
fig.savefig(out, dpi=70)
print(out, [int(r.img_id) for _, r in sel])
