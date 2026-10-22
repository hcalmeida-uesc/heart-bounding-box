"""Caminho 2, Passo 1 (extração): só a detecção de círculos, sem Level Set, sem rótulo.

Por caso, grava em c2_cache/<id>.npz:
- asc: círculos da ascendente pelo rastreamento de produção (detect_aorta_circles, config de produção),
  colunas z, c0 (eixo 0), c1 (eixo 1), r, accum, interp, mean_hu, std_hu, hole (disco de 0,7 r;
  hole = fração do disco em -300, excluída da média e do desvio);
- desc: candidatos à descendente por fatia com ascendente detectada (não interpolada): Hough com faixa
  de raio própria (5 a 14 mm) restrita às colunas posteriores ao centro da ascendente (eixo 1 menor),
  colunas z, c0, c1, r, accum, mean_hu, std_hu, hole.
Coordenadas em voxels da resolução reduzida (fator 2 em eixo 0 e 1). O label não entra nas
características: load_and_preprocess_image o carrega junto com a imagem, e ele é descartado aqui."""
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.aorta_localization import detect_aorta_circles  # noqa: E402
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "c2_cache"
ORIG = Path("D:/Doutorado/dataset/original")
CFG = json.loads((REPO / "config/pipeline_config.json").read_text())
DESC_R_MM = (5.0, 14.0)
DESC_PEAKS = 25


def block_of(i):
    lo = (i - 1) // 200 * 200 + 1
    return f"{lo}-{lo + 199}"


def disk_stats(s, c0, c1, r, frac=0.7):
    rr = max(1.0, frac * r)
    a0, a1 = np.ogrid[:s.shape[0], :s.shape[1]]
    m = (a0 - c0) ** 2 + (a1 - c1) ** 2 <= rr ** 2
    v = s[m]
    # -300 = fora do limiar (inclui os "buracos" acima do p99,7, comuns no lúmen com muito contraste)
    w = v[v > -299.5]
    hole = 1.0 - w.size / max(v.size, 1)
    return (float(w.mean()) if w.size else np.nan, float(w.std()) if w.size else np.nan, hole)


def work(i):
    out = CACHE / f"{i}.npz"
    d = load_and_preprocess_image(i, str(ORIG / block_of(i)), CFG)
    im = d["lcc_image"]
    del d["label"]
    sp = d["scaled_spacing"]
    psp = float(sp[0] + sp[1]) / 2.0
    cc = CFG["CIRCLE_DETECTION"]
    circ = detect_aorta_circles(
        im, np.arange(cc["radii_start_px"], cc["radii_end_px"], cc.get("radius_step_px", 1)), psp,
        tol_radius_mm=cc["tol_radius_mm"], tol_distance_mm=cc["tol_distance_mm"],
        quadrant_offset=tuple(cc["quadrant_offset"]), max_slice_miss_threshold=cc["max_slice_miss_threshold"],
        neighbor_distance_threshold=cc["neighbor_distance_threshold"],
        total_num_peaks_initial=cc["total_num_peaks_initial"], total_num_peaks=cc["total_num_peaks"],
        canny_sigma=cc["canny_sigma"], use_local_roi=cc["use_local_roi"], local_roi_padding=cc["local_roi_padding"],
        out_of_tolerance_as_miss=cc["out_of_tolerance_as_miss"],
        interpolate_missed_circles=cc["interpolate_missed_circles"],
        candidate_selection_strategy=cc["candidate_selection_strategy"], verbose=False)
    radii = np.arange(int(np.floor(DESC_R_MM[0] / psp)), int(np.ceil(DESC_R_MM[1] / psp)) + 1)
    asc, desc = [], []
    for c in circ:
        z = int(c["slice_index"])
        s = im[:, :, z]
        # convenção do skimage: center_y = linha = eixo 0; center_x = coluna = eixo 1
        c0, c1, r = float(c["center_y"]), float(c["center_x"]), float(c["radius"])
        interp = bool(c.get("interpolated", False))
        asc.append([z, c0, c1, r, float(c.get("accum", np.nan)), float(interp), *disk_stats(s, c0, c1, r)])
        if interp:
            continue
        hi1 = int(np.floor(c1))
        if hi1 < 2 * radii[0] + 2:
            continue
        roi = s[:, :hi1]
        edges = feature.canny(roi.astype(float), sigma=cc["canny_sigma"])
        h = hough_circle(edges, radii)
        acc, x1, x0, rr = hough_circle_peaks(h, radii, min_xdistance=4, min_ydistance=4,
                                             total_num_peaks=DESC_PEAKS)
        for a_, p1, p0, r_ in zip(acc, x1, x0, rr):
            desc.append([z, float(p0), float(p1), float(r_), float(a_), *disk_stats(s, p0, p1, r_)])
    np.savez_compressed(out, asc=np.array(asc, np.float32).reshape(-1, 9),
                        desc=np.array(desc, np.float32).reshape(-1, 8))
    return dict(img_id=i, shape0=im.shape[0], shape1=im.shape[1], shape2=im.shape[2],
                rsp0=float(sp[0]), rsp1=float(sp[1]), rsp2=float(sp[2]),
                fsp0=float(d["spacing"][0]), fsp1=float(d["spacing"][1]), fsp2=float(d["spacing"][2]),
                n_asc=len(asc), n_asc_real=int(sum(1 for a in asc if a[5] == 0)), n_desc_cand=len(desc),
                desc_r_px_min=int(radii[0]), desc_r_px_max=int(radii[-1]))


def safe(i):
    try:
        return work(i)
    except Exception as e:  # noqa: BLE001
        return dict(img_id=i, error=repr(e))


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    ids = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 1001))
    done = {int(p.stem) for p in CACHE.glob("*.npz")}
    meta_p = HERE / "c2_extract_meta.csv"
    prev = pd.read_csv(meta_p) if meta_p.exists() else pd.DataFrame()
    todo = [i for i in ids if i not in done]
    print("a processar:", len(todo), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for k, r in enumerate(ex.map(safe, todo, chunksize=1)):
            rows.append(r)
            if k % 50 == 0:
                print(k, r, flush=True)
    allm = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates("img_id", keep="last").sort_values("img_id")
    allm.to_csv(meta_p, index=False)
    print("ok", len(rows))
