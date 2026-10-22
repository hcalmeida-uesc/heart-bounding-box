"""Caminho 2, Passo 1 (complemento): re-mede o raio da ascendente na janela da raiz com faixa larga.

A faixa de produção (18 a 30 px reduzidos) impõe diâmetro mínimo de ~25 a 28 mm. Aqui, nas fatias da
janela da raiz (0 a 20 mm acima do menor z rastreado), roda a Hough com raio de 6 a 22 mm numa ROI em
torno do centro rastreado e fica com o pico de maior acumulador cujo centro está a até 6 mm do centro
rastreado. Sem rótulo. Saída: c2_asc_wide/<id>.npy com colunas z, c0, c1, r, accum (voxels reduzidos)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from skimage import feature  # noqa: E402
from skimage.transform import hough_circle, hough_circle_peaks  # noqa: E402

from c2_extract import CFG, ORIG, block_of, load_and_preprocess_image  # noqa: E402
from c2_features import ROOT_WIN_MM  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "c2_asc_wide"
R_MM = (6.0, 22.0)
MAX_SHIFT_MM = 6.0


def work(i):
    asc = np.load(HERE / f"c2_cache/{i}.npz")["asc"]
    if len(asc) == 0:
        np.save(OUT / f"{i}.npy", np.zeros((0, 5), np.float32))
        return i
    d = load_and_preprocess_image(i, str(ORIG / block_of(i)), CFG)
    im, sp = d["lcc_image"], d["scaled_spacing"]
    psp = float(sp[0] + sp[1]) / 2
    real = asc[asc[:, 5] == 0]
    win = real[real[:, 0] <= asc[:, 0].min() + ROOT_WIN_MM / float(sp[2])]
    radii = np.arange(int(np.floor(R_MM[0] / psp)), int(np.ceil(R_MM[1] / psp)) + 1)
    half = int(np.ceil(radii[-1] + MAX_SHIFT_MM / psp + 6))
    rows = []
    for z, c0, c1 in win[:, :3]:
        a0, a1 = max(0, int(c0) - half), max(0, int(c1) - half)
        roi = im[a0:int(c0) + half + 1, a1:int(c1) + half + 1, int(z)]
        h = hough_circle(feature.canny(roi.astype(float), sigma=CFG["CIRCLE_DETECTION"]["canny_sigma"]), radii)
        acc, x1, x0, rr = hough_circle_peaks(h, radii, min_xdistance=2, min_ydistance=2, total_num_peaks=30)
        ok = np.hypot(x0 + a0 - c0, x1 + a1 - c1) <= MAX_SHIFT_MM / psp
        if ok.any():
            k = np.where(ok)[0][np.argmax(acc[ok])]
            rows.append([z, x0[k] + a0, x1[k] + a1, rr[k], acc[k]])
    np.save(OUT / f"{i}.npy", np.array(rows, np.float32).reshape(-1, 5))
    return i


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    done = {int(p.stem) for p in OUT.glob("*.npy")}
    todo = sorted(int(p.stem) for p in (HERE / "c2_cache").glob("*.npz") if int(p.stem) not in done)
    print("a processar:", len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 24) as ex:
        list(ex.map(work, todo, chunksize=2))
    print("ok")
