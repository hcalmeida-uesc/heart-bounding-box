"""Passo 0 bbox dinâmico: extrai, por caso (624 = interseção - teste), coordenadas dos voxels do
label (resolução integral), shape, spacing (cabeçalho NIfTI) e bbox Opção 1/2 recalculados.
Só leitura; nada de produção é tocado."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

OUT = Path(__file__).parent / "p0_cache"
ORIG = Path("D:/Doutorado/dataset/original")
MANIFEST = Path("D:/Doutorado/dataset/cropped_manifests/manifest_all.csv")
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())
SPLIT = Path("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")


def work(args):
    img_id, block, ay, ax, az = args
    nii = nib.load(str(ORIG / block / f"{img_id}.label.nii.gz"))
    lab = np.asarray(nii.dataobj)
    ys, xs, zs = np.nonzero(lab == 1)
    shape = lab.shape
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    anchor = (ay, ax, az)
    b1 = compute_full_resolution_bbox(anchor, CAL["option1"]["offset_min"], CAL["option1"]["offset_max"],
                                      CAL["downscale_factors"], shape)
    b2 = compute_full_resolution_bbox(anchor, CAL["option2"]["offset_min"], CAL["option2"]["offset_max"],
                                      CAL["downscale_factors"], shape)
    np.savez_compressed(OUT / f"{img_id}.npz", y=ys.astype(np.uint16), x=xs.astype(np.uint16),
                        z=zs.astype(np.uint16))
    return dict(img_id=img_id, block=block, sy=shape[0], sx=shape[1], sz=shape[2],
                spy=spacing[0], spx=spacing[1], spz=spacing[2], n_label=len(ys),
                anchor_y=ay, anchor_x=ax, anchor_z=az,
                b1_min_y=b1[0][0], b1_min_x=b1[0][1], b1_min_z=b1[0][2],
                b1_max_y=b1[1][0], b1_max_x=b1[1][1], b1_max_z=b1[1][2],
                b2_min_y=b2[0][0], b2_min_x=b2[0][1], b2_min_z=b2[0][2],
                b2_max_y=b2[1][0], b2_max_x=b2[1][1], b2_max_z=b2[1][2])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    m = pd.read_csv(MANIFEST)
    so = pd.read_csv(SPLIT)
    test = set(so.loc[so.split == "test", "ID"])
    inter = m[(m.option1_saved == True) & (m.option2_saved == True)]  # noqa: E712
    cases = inter[~inter.img_id.isin(test)]
    assert len(cases) == 624, len(cases)
    ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv").set_index("img_id")
    jobs = [(int(r.img_id), r.block, float(ac.at[r.img_id, "anchor_y"]), float(ac.at[r.img_id, "anchor_x"]),
             float(ac.at[r.img_id, "anchor_z"])) for r in cases.itertuples()]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, row in enumerate(ex.map(work, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows).merge(m[["img_id", "option1_loss_pct", "option2_loss_pct"]], on="img_id")
    df.to_csv(OUT.parent / "p0_cases.csv", index=False)
    print("ok", len(df))
