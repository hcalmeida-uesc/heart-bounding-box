"""Passo 0b bbox dinâmico: comparação de âncoras de posição sem rótulo (A1-A5).

Só leitura. Insumos: p0_cases.csv / p0_percase.csv (Passo 0), all_cases.csv, candidate_anchor_z.csv
(A5, amostra), cache de coordenadas do label do Passo 0 (p0_cache/, regenerável por p0_extract.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.aorta_anchor import select_cases_by_circle_count_percentile  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "p0_cache"
P0 = REPO / "output/bounding_box/dynamic_bbox_step0"
FACT = np.array([2.0, 2.0, 1.0])
AX = ("y", "x", "z")

cases = pd.read_csv(P0 / "p0_cases.csv").sort_values("img_id").reset_index(drop=True)
percase0 = pd.read_csv(P0 / "p0_percase.csv").set_index("img_id")
all_cases = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv")
cases = cases.merge(all_cases[["img_id", "num_circles_detected", "aorta_voxels"]], on="img_id")
cand = pd.read_csv(REPO / "output/bounding_box/aorta_anchor_z_diagnosis/candidate_anchor_z.csv")
cand = cand[cand.status == "ok"].set_index("img_id")

raw = {}
for i in cases.img_id:
    d = np.load(CACHE / f"{i}.npz")
    raw[i] = np.stack([d[a].astype(np.int32) for a in AX], axis=1)
lab_min = np.array([raw[i].min(axis=0) for i in cases.img_id], float)
lab_max = np.array([raw[i].max(axis=0) for i in cases.img_id], float)
cases[[f"lab_ctr_{a}" for a in AX]] = (lab_min + lab_max + 1) / 2
cases[[f"lab_ext_{a}" for a in AX]] = lab_max - lab_min + 1


def arr(df, prefix):
    return df[[f"{prefix}{a}" for a in AX]].to_numpy(float)


def anchors(df):
    shape = arr(df, "s")
    a1 = arr(df, "anchor_") * FACT
    frac = np.median(arr(df, "lab_ctr_") / shape, axis=0)
    a3 = frac * shape
    out = {"A1 aorta (atual)": a1, "A2 centro do volume": shape / 2.0, "A3 fração relativa fixa": a3,
           "A4 mista (y,x de A1; z de A3)": np.column_stack([a1[:, 0], a1[:, 1], a3[:, 2]])}
    if df.img_id.isin(cand.index).all():
        zmin = cand.loc[df.img_id, "z_min_candidate"].to_numpy(float)
        out["A5 z mínimo da aorta (y,x de A1)"] = np.column_stack([a1[:, 0], a1[:, 1], zmin])
    return out, frac


def evaluate(df, lo, hi):
    shape = arr(df, "s")
    lo = np.clip(np.floor(lo), 0, shape)
    hi = np.clip(np.ceil(hi), 0, shape)
    rv = 1.0 - np.maximum(hi - lo, 0).prod(axis=1) / shape.prod(axis=1)
    loss = np.array([1.0 - np.all((raw[i] >= lo[k]) & (raw[i] < hi[k]), axis=1).mean()
                     for k, i in enumerate(df.img_id)])
    return rv * 100, loss * 100


PERCASE = {}


def run(df, label):
    ctr = arr(df, "lab_ctr_")
    ext = arr(df, "lab_ext_")
    sp = arr(df, "sp")
    anc, frac = anchors(df)
    err_rows, box_rows = [], []
    for name, a in anc.items():
        err = ctr - a
        res_v = np.abs(err - np.median(err, axis=0))
        err_mm = err * sp
        res_mm = np.abs(err_mm - np.median(err_mm, axis=0))
        for j, ax in enumerate(AX):
            err_rows.append(dict(conjunto=label, ancora=name, eixo=ax,
                                 vies_vox=float(np.median(err[:, j])), vies_mm=float(np.median(err_mm[:, j])),
                                 med_vox=float(np.median(res_v[:, j])), p95_vox=float(np.percentile(res_v[:, j], 95)),
                                 max_vox=float(res_v[:, j].max()), med_mm=float(np.median(res_mm[:, j])),
                                 p95_mm=float(np.percentile(res_mm[:, j], 95)), max_mm=float(res_mm[:, j].max())))
        pred = a + np.median(err, axis=0)
        for pct in (95, 100):
            m = np.percentile(res_v, pct, axis=0)
            rv, loss = evaluate(df, pred - ext / 2 - m, pred + ext / 2 + m)
            if label == "todos (624)":
                PERCASE[f"loss_{name.split()[0]}_p{pct}"] = loss
                if pct == 95:
                    for j, ax in enumerate(AX):
                        PERCASE[f"resid_{name.split()[0]}_{ax}_vox"] = err[:, j] - np.median(err[:, j])
            box_rows.append(dict(conjunto=label, ancora=name, pct=pct, rv_mean=rv.mean(), rv_median=np.median(rv),
                                 loss_mean=loss.mean(), loss_max=loss.max(), n_loss_pos=int((loss > 1e-9).sum()),
                                 margin_y=m[0], margin_x=m[1], margin_z=m[2]))
    # referências no mesmo conjunto: piso v1, oráculo de posição (v6) recalculado, teto v4
    for pct in (95, 100):
        size = np.percentile(ext, pct, axis=0)
        rv, loss = evaluate(df, ctr - size / 2, ctr + size / 2)
        box_rows.append(dict(conjunto=label, ancora="REF oráculo de posição (v6)", pct=pct, rv_mean=rv.mean(),
                             rv_median=np.median(rv), loss_mean=loss.mean(), loss_max=loss.max(),
                             n_loss_pos=int((loss > 1e-9).sum())))
    p0 = percase0.loc[df.img_id]
    refs = dict(v1_rv=p0.rv_1.mean(), v1_loss=p0.loss_1.mean(), v4_rv=p0.rv_4.mean(), v3_rv=p0.rv_3.mean(),
                a3_frac=frac.tolist(), n=len(df))
    return pd.DataFrame(err_rows), pd.DataFrame(box_rows), refs


# Critério A literal (limiares globais sobre os 994 ok) — reportado; exclui 0 dos 624.
excluded_a, _ = select_cases_by_circle_count_percentile(all_cases, 5.0)
lo_a, hi_a = np.percentile(all_cases.loc[all_cases.status == "ok", "num_circles_detected"], [2.5, 97.5])
n_excl_global = int(cases.img_id.isin(excluded_a.img_id).sum())
# Variante informativa: mesma regra (5% total, 2,5% por cauda) aplicada dentro dos 624.
cases_ok = cases.assign(status="ok")
excluded_in, _ = select_cases_by_circle_count_percentile(cases_ok, 5.0)
lo_in, hi_in = np.percentile(cases.num_circles_detected, [2.5, 97.5])
keep = ~cases.img_id.isin(excluded_in.img_id)
sub_a5 = cases.img_id.isin(cand.index)

sets = [(cases, "todos (624)"),
        (cases[keep].reset_index(drop=True), f"sem 5% extremos de num_circles nos 624 ({int(keep.sum())})"),
        (cases[sub_a5].reset_index(drop=True), f"amostra A5 ({int(sub_a5.sum())})")]
errs, boxes, refs = [], [], {}
for df, label in sets:
    e, b, r = run(df, label)
    errs.append(e)
    boxes.append(b)
    refs[label] = r
errs = pd.concat(errs)
boxes = pd.concat(boxes)
for label, r in refs.items():
    gap = r["v4_rv"] - r["v1_rv"]
    m = boxes.conjunto == label
    boxes.loc[m, "frac_gap"] = (boxes.loc[m, "rv_mean"] - r["v1_rv"]) / gap

pd.set_option("display.width", 250)
print("limiares crit. A (5%, global 994 ok):", lo_a, hi_a, " excluídos entre os 624:", n_excl_global)
print("limiares 5% dentro dos 624:", lo_in, hi_in, " excluídos:", int((~keep).sum()),
      sorted(cases.img_id[~keep].tolist()))
print(json.dumps(refs, indent=1, default=float))
print(errs.round(2).to_string(index=False))
print(boxes.round(3).to_string(index=False))
a5_overlap = cases[sub_a5]
print("amostra A5: num_circles quintis/distr:", a5_overlap.num_circles_detected.describe().round(1).to_dict(),
      " excluídos crit A na amostra:", int((~keep[sub_a5]).sum()))

OUT = REPO / "output/bounding_box/dynamic_bbox_step0b"
OUT.mkdir(exist_ok=True)
pc = pd.DataFrame({"img_id": cases.img_id, "num_circles_detected": cases.num_circles_detected, **PERCASE})
pc.to_csv(OUT / "p0b_percase.csv", index=False)
print("top perda p95:")
for a in ("A1", "A2", "A4"):
    top = pc.sort_values(f"loss_{a}_p95", ascending=False).head(4)
    print(a, [(int(r.img_id), round(getattr(r, f"loss_{a}_p95"), 1), int(r.num_circles_detected))
              for r in top.itertuples()])
errs.to_csv(OUT / "p0b_anchor_errors.csv", index=False)
boxes.to_csv(OUT / "p0b_anchor_boxes.csv", index=False)
(OUT / "p0b_meta.json").write_text(json.dumps(dict(refs=refs, crit_a_global_thresholds=[lo_a, hi_a],
                                                   crit_a_global_excluded_in_624=n_excl_global,
                                                   within624_thresholds=[lo_in, hi_in],
                                                   within624_excluded_ids=sorted(cases.img_id[~keep].tolist()),
                                                   a5_sample_ids=sorted(cases.img_id[sub_a5].tolist())),
                                              indent=2, default=float))
