"""Refaz o cache de coordenadas do label (p0_cache/) para os 1000 casos. Só lê labels NIfTI."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "p0_cache"
ORIG = Path("D:/Doutorado/dataset/original")


def block_of(i):
    lo = (i - 1) // 200 * 200 + 1
    return f"{lo}-{lo + 199}"


def work(i):
    lab = np.asarray(nib.load(str(ORIG / block_of(i) / f"{i}.label.nii.gz")).dataobj)
    ys, xs, zs = np.nonzero(lab == 1)
    np.savez_compressed(OUT / f"{i}.npz", y=ys.astype(np.uint16), x=xs.astype(np.uint16), z=zs.astype(np.uint16))
    return i


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        n = len(list(ex.map(work, range(1, 1001), chunksize=4)))
    print("ok", n)
