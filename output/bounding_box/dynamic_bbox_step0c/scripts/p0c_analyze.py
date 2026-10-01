"""Passo 0c bbox dinâmico: bbox de tamanho FIXO posicionado por âncoras diferentes (A1, A2, A6),
p calibrado para igualar a perda média da Opção 1 no mesmo conjunto. Só leitura."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "p0_cache"
FACT = np.array([2.0, 2.0, 1.0])
AX = ("y", "x", "z")
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())

so = pd.read_csv("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")
ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv")
man = pd.read_csv("D:/Doutorado/dataset/cropped_manifests/manifest_all.csv")
c624 = pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0/p0_cases.csv")
extra = pd.read_csv(HERE / "p0c_extra_cases.csv")
cols = ["img_id", "block", "sy", "sx", "sz", "spy", "spx", "spz"]
cases = pd.concat([c624[cols], extra[cols]]).drop_duplicates("img_id")
cases = cases[cases.img_id.isin(so.loc[so.split != "test", "ID"])]
cases = cases.merge(ac[["img_id", "status", "anchor_y", "anchor_x", "anchor_z", "aorta_voxels",
                        "num_circles_detected"]], on="img_id", how="left")
cases = cases.merge(man[["img_id", "in_option1_population"]], on="img_id", how="left")
cases = cases.sort_values("img_id").reset_index(drop=True)
assert len(cases) == 792, len(cases)

raw = {}
for i in cases.img_id:
    d = np.load(CACHE / f"{i}.npz")
    raw[i] = np.stack([d[a].astype(np.int32) for a in AX], axis=1)
cases[[f"lmin_{a}" for a in AX]] = np.array([raw[i].min(axis=0) for i in cases.img_id], float)
cases[[f"lmax_{a}" for a in AX]] = np.array([raw[i].max(axis=0) for i in cases.img_id], float)


def arr(df, prefix):
    return df[[f"{prefix}{a}" for a in AX]].to_numpy(float)


def anchor(df, name):
    shape = arr(df, "s")
    if name == "A2 centro do volume":
        return shape / 2.0
    a1 = arr(df, "anchor_") * FACT
    if name == "A1 aorta (produção)":
        return a1
    return np.column_stack([shape[:, 0] / 2.0, a1[:, 1], shape[:, 2] / 2.0])  # A6


def evaluate(df, lo, hi):
    shape = arr(df, "s")
    lo = np.clip(np.floor(lo), 0, shape)
    hi = np.clip(np.ceil(hi), 0, shape)
    rv = 1.0 - np.maximum(hi - lo, 0).prod(axis=1) / shape.prod(axis=1)
    loss = np.array([1.0 - np.all((raw[i] >= lo[k]) & (raw[i] < hi[k]), axis=1).mean()
                     for k, i in enumerate(df.img_id)])
    return rv * 100, loss * 100


def option1(df):
    lo, hi = [], []
    for r in df.itertuples():
        b0, b1 = compute_full_resolution_bbox((r.anchor_y, r.anchor_x, r.anchor_z), CAL["option1"]["offset_min"],
                                              CAL["option1"]["offset_max"], CAL["downscale_factors"],
                                              (int(r.sy), int(r.sx), int(r.sz)))
        lo.append(b0)
        hi.append(b1)
    return evaluate(df, np.array(lo, float), np.array(hi, float))


def fixed_box(df, a, p):
    d_lo = arr(df, "lmin_") - a
    d_hi = arr(df, "lmax_") + 1 - a
    off_lo = np.percentile(d_lo, 100 - p, axis=0)
    off_hi = np.percentile(d_hi, p, axis=0)
    return a + off_lo, a + off_hi, off_hi - off_lo


def calibrate(df, a, target):
    lo_p, hi_p = 50.0, 100.0
    for _ in range(32):
        mid = (lo_p + hi_p) / 2
        lo, hi, _ = fixed_box(df, a, mid)
        if evaluate(df, lo, hi)[1].mean() > target:
            lo_p = mid
        else:
            hi_p = mid
    p = hi_p  # menor p testado com perda média <= alvo
    lo, hi, size = fixed_box(df, a, p)
    rv, loss = evaluate(df, lo, hi)
    return p, size, rv, loss


def stats_row(label, anc, n, p, size, rv, loss, df, target):
    top = np.argsort(-loss)[:5]
    sp_med = np.median(arr(df, "sp"), axis=0)
    return dict(conjunto=label, ancora=anc, n=n, p=p,
                caixa_vox="x".join(f"{s:.1f}" for s in size),
                caixa_mm_med="x".join(f"{s * m:.1f}" for s, m in zip(size, sp_med)),
                rv_mean=rv.mean(), rv_median=np.median(rv), rv_p5=np.percentile(rv, 5), rv_p95=np.percentile(rv, 95),
                loss_target=target, loss_mean=loss.mean(), loss_max=loss.max(), n_loss_pos=int((loss > 1e-9).sum()),
                top5="; ".join(f"{int(df.img_id.iloc[k])}:{loss[k]:.1f}%(nc={df.num_circles_detected.iloc[k]:.0f},"
                               f"C={'s' if df.in_option1_population.iloc[k] else 'n'})" for k in top))


set1 = cases[cases.img_id.isin(c624.img_id)].reset_index(drop=True)
set2a = cases[cases.status == "ok"].reset_index(drop=True)
set2b = cases.reset_index(drop=True)
assert len(set1) == 624 and len(set2a) == 786

rows, percase = [], {}
for label, df, anchors_ in (("1: 624 casos do Passo 0", set1, ("A1 aorta (produção)", "A2 centro do volume", "A6 mista")),
                            ("2: fora do teste, aorta detectada (786)", set2a,
                             ("A1 aorta (produção)", "A2 centro do volume", "A6 mista"))):
    rv1, loss1 = option1(df)
    target = loss1.mean()
    size1 = np.array([CAL["option1"]["bbox_extent_full_resolution"][a] for a in AX])
    rows.append(stats_row(label, "Opção 1 (produção)", len(df), float("nan"), size1, rv1, loss1, df, target))
    percase[(label, "Opção 1")] = (df.img_id.to_numpy(), rv1, loss1)
    for anc in anchors_:
        p, size, rv, loss = calibrate(df, anchor(df, anc), target)
        rows.append(stats_row(label, anc, len(df), p, size, rv, loss, df, target))
        percase[(label, anc)] = (df.img_id.to_numpy(), rv, loss)
        print(label, anc, "p=", p, flush=True)
    if label.startswith("2"):
        target2 = target
# conjunto 2b: A2 em todos os 792 (alvo = perda média da Opção 1 nos 786, único conjunto onde ela existe)
p, size, rv, loss = calibrate(set2b, anchor(set2b, "A2 centro do volume"), target2)
rows.append(stats_row("2b: fora do teste, todos (792)", "A2 centro do volume", len(set2b), p, size, rv, loss, set2b,
                      target2))
percase[("2b", "A2")] = (set2b.img_id.to_numpy(), rv, loss)
fail = set2b.status != "ok"
print("2b: perdas nos 6 casos sem aorta:", dict(zip(set2b.img_id[fail], np.round(loss[fail.to_numpy()], 2))),
      " Rv:", np.round(rv[fail.to_numpy()], 1))

out = pd.DataFrame(rows)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 200)
print(out.drop(columns=["top5"]).round(4).to_string(index=False))
for r in out.itertuples():
    print(r.conjunto, "|", r.ancora, "|", r.top5)

OUTDIR = REPO / "output/bounding_box/dynamic_bbox_step0c"
OUTDIR.mkdir(exist_ok=True)
out.to_csv(OUTDIR / "p0c_fixed_box_summary.csv", index=False)
pcs = []
for (label, anc), (ids, rv, loss) in percase.items():
    pcs.append(pd.DataFrame(dict(conjunto=label, ancora=anc, img_id=ids, rv=rv, loss=loss)))
pc = pd.concat(pcs).merge(cases[["img_id", "num_circles_detected", "aorta_voxels", "in_option1_population"]],
                          on="img_id")
pc.to_csv(OUTDIR / "p0c_percase.csv", index=False)
# decomposição do conjunto 2a: casos dentro vs fora da população do Critério C (Opção 1)
m = pc[pc.conjunto.str.startswith("2:")]
print(m.groupby(["ancora", "in_option1_population"]).agg(n=("loss", "size"), rv=("rv", "mean"),
                                                        loss=("loss", "mean"), loss_max=("loss", "max"),
                                                        n_pos=("loss", lambda s: int((s > 1e-9).sum()))).round(3))
