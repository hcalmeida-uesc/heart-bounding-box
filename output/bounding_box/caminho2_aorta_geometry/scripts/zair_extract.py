"""Zair et al. (2022), Seção III-A/B: medidas das duas aortas numa única fatia (sem rótulo).

O artigo escolhe a "first slice" (bifurcação do tronco pulmonar, acima das coronárias) por registro
com um atlas e Dice. Sem atlas, a aproximação aqui é a fatia mais cranial em que a ascendente
(rastreamento de produção) e a descendente (seleção de c2_select) são detectadas. Nessa fatia:
- ascendente: Hough de novo com raio de 6 a 22 mm, numa ROI em torno do centro rastreado, pico de
  maior acumulador a até 6 mm do centro rastreado (a faixa de produção trunca o diâmetro);
- descendente: o candidato escolhido naquela fatia (Hough com raio de 5 a 14 mm).
Saída: zair_slice.csv (voxels reduzidos e mm)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from skimage import feature  # noqa: E402
from skimage.transform import hough_circle, hough_circle_peaks  # noqa: E402

from c2_extract import CFG, ORIG, block_of, load_and_preprocess_image  # noqa: E402
from c2_select import MIN_SLICES, select_desc  # noqa: E402

HERE = Path(__file__).parent
R_MM = (6.0, 22.0)
MAX_SHIFT_MM = 6.0


def work(i):
    meta = pd.read_csv(HERE / "c2_extract_meta.csv").set_index("img_id").loc[i]
    d = np.load(HERE / f"c2_cache/{i}.npz")
    asc, desc = d["asc"], d["desc"]
    out = dict(img_id=i, zair_ok=False)
    if len(asc) == 0:
        return out
    ch, _, _ = select_desc(asc, desc, float(meta.rsp0))
    if len(ch) < MIN_SLICES:
        return out
    z = int(ch[:, 0].max())
    a = asc[asc[:, 0] == z][0]
    q = ch[ch[:, 0] == z][0]
    im = load_and_preprocess_image(i, str(ORIG / block_of(i)), CFG)
    sp = im["scaled_spacing"]
    psp = float(sp[0] + sp[1]) / 2
    s = im["lcc_image"][:, :, z]
    radii = np.arange(int(np.floor(R_MM[0] / psp)), int(np.ceil(R_MM[1] / psp)) + 1)
    half = int(np.ceil(radii[-1] + MAX_SHIFT_MM / psp + 6))
    c0, c1 = float(a[1]), float(a[2])
    a0, a1 = max(0, int(c0) - half), max(0, int(c1) - half)
    roi = s[a0:int(c0) + half + 1, a1:int(c1) + half + 1]
    h = hough_circle(feature.canny(roi.astype(float), sigma=CFG["CIRCLE_DETECTION"]["canny_sigma"]), radii)
    acc, x1, x0, rr = hough_circle_peaks(h, radii, min_xdistance=2, min_ydistance=2, total_num_peaks=30)
    ok = np.hypot(x0 + a0 - c0, x1 + a1 - c1) <= MAX_SHIFT_MM / psp
    if ok.any():
        k = np.where(ok)[0][np.argmax(acc[ok])]
        ac0, ac1, ar = float(x0[k] + a0), float(x1[k] + a1), float(rr[k])
        wide = True
    else:
        ac0, ac1, ar, wide = c0, c1, float(a[3]), False
    out.update(zair_ok=True, z_sel=z, z_top=int(asc[:, 0].max()), asc_wide_found=wide,
               asc_c0_red=ac0, asc_c1_red=ac1, asc_r_red=ar, ascP_r_red=float(a[3]),
               desc_c0_red=float(q[1]), desc_c1_red=float(q[2]), desc_r_red=float(q[3]), rsp=psp)
    return out


if __name__ == "__main__":
    ids = sorted(int(p.stem) for p in (HERE / "c2_cache").glob("*.npz"))
    with ProcessPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(work, ids, chunksize=4))
    pd.DataFrame(rows).to_csv(HERE / "zair_slice.csv", index=False)
    print("ok", len(rows))
