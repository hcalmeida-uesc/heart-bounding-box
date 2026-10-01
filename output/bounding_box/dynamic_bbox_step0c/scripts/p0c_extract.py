"""Passo 0c: estende o cache de coordenadas do label (p0_cache/) aos casos fora do teste que o
Passo 0 não cobriu (792 - 624 = 168). Só lê labels NIfTI; não roda segmentação."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "p0_cache"
ORIG = Path("D:/Doutorado/dataset/original")
REPO = Path("D:/Doutorado/heart-bounding-box")
SPLIT = Path("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")
AX = ("y", "x", "z")


def block_of(img_id):
    lo = (img_id - 1) // 200 * 200 + 1
    return f"{lo}-{lo + 199}"


def work(img_id):
    block = block_of(img_id)
    nii = nib.load(str(ORIG / block / f"{img_id}.label.nii.gz"))
    lab = np.asarray(nii.dataobj)
    ys, xs, zs = np.nonzero(lab == 1)
    np.savez_compressed(OUT / f"{img_id}.npz", y=ys.astype(np.uint16), x=xs.astype(np.uint16),
                        z=zs.astype(np.uint16))
    sp = nii.header.get_zooms()[:3]
    return dict(img_id=img_id, block=block, sy=lab.shape[0], sx=lab.shape[1], sz=lab.shape[2],
                spy=float(sp[0]), spx=float(sp[1]), spz=float(sp[2]), n_label=len(ys))


if __name__ == "__main__":
    so = pd.read_csv(SPLIT)
    nontest = set(so.loc[so.split != "test", "ID"])
    done = set(pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0/p0_cases.csv").img_id)
    todo = sorted(nontest - done)
    print("a extrair:", len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(work, todo, chunksize=4))
    pd.DataFrame(rows).to_csv(HERE / "p0c_extra_cases.csv", index=False)
    print("ok", len(rows))
