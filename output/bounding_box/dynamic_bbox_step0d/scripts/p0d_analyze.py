"""Passo 0d bbox dinâmico. Parte A: curvas Rv x risco de perda por método (792 fora do teste).
Parte B: FOV descentralizado simulado (624 casos). Só leitura; depende de p0_cache/."""
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

HERE = Path(__file__).parent
CACHE = HERE / "p0_cache"
OUTDIR = REPO / "output/bounding_box/dynamic_bbox_step0d"
FACT = np.array([2.0, 2.0, 1.0])
AX = ("y", "x", "z")
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())
P_GRID = np.round(np.arange(80.0, 100.0001, 0.5), 1)

RAW = {}


def load_cases():
    so = pd.read_csv("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")
    ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv")
    man = pd.read_csv("D:/Doutorado/dataset/cropped_manifests/manifest_all.csv")
    c624 = pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0/p0_cases.csv")
    extra = pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0c/p0c_extra_cases.csv")
    cols = ["img_id", "block", "sy", "sx", "sz", "spy", "spx", "spz"]
    cases = pd.concat([c624[cols], extra[cols]]).drop_duplicates("img_id")
    cases = cases[cases.img_id.isin(so.loc[so.split != "test", "ID"])]
    cases = cases.merge(ac[["img_id", "status", "anchor_y", "anchor_x", "anchor_z"]], on="img_id", how="left")
    cases = cases.merge(man[["img_id", "in_option1_population"]], on="img_id", how="left")
    cases["in_c"] = cases.in_option1_population.fillna(False).astype(bool)
    cases["aorta_ok"] = cases.status == "ok"
    cases["in_624"] = cases.img_id.isin(c624.img_id)
    return cases.sort_values("img_id").reset_index(drop=True)


def init_worker(ids):
    for i in ids:
        d = np.load(CACHE / f"{i}.npz")
        RAW[i] = np.stack([d[a].astype(np.int16) for a in AX], axis=1)


def arr(df, prefix):
    return df[[f"{prefix}{a}" for a in AX]].to_numpy(float)


def anchor(df, name):
    shape = arr(df, "s")
    a2 = shape / 2.0
    if name == "A2":
        return a2
    a1 = arr(df, "anchor_") * FACT
    if name == "A1":
        return a1
    x = np.where(np.isnan(a1[:, 1]), a2[:, 1], a1[:, 1])  # A6; sem aorta -> x de A2
    return np.column_stack([a2[:, 0], x, a2[:, 2]])


def evaluate(ids, shape, lo, hi, shift=None):
    lo = np.clip(np.floor(lo), 0, shape)
    hi = np.clip(np.ceil(hi), 0, shape)
    rv = 1.0 - np.maximum(hi - lo, 0).prod(axis=1) / shape.prod(axis=1)
    loss = np.empty(len(ids))
    for k, i in enumerate(ids):
        p = RAW[i] if shift is None else RAW[i] - shift
        loss[k] = 1.0 - np.all((p >= lo[k]) & (p < hi[k]), axis=1).mean()
    return rv * 100, loss * 100


def offsets(df, a, p):
    d_lo = arr(df, "lmin_") - a
    d_hi = arr(df, "lmax_") + 1 - a
    return np.percentile(d_lo, 100 - p, axis=0), np.percentile(d_hi, p, axis=0)


def summarize(rv, loss):
    return dict(rv_mean=rv.mean(), rv_median=np.median(rv), loss_mean=loss.mean(),
                frac_gt0=(loss > 1e-9).mean() * 100, frac_gt1=(loss > 1).mean() * 100,
                frac_gt5=(loss > 5).mean() * 100, frac_gt20=(loss > 20).mean() * 100, loss_max=loss.max(),
                n_gt1=int((loss > 1).sum()))


METHODS = {
    "M1 A1 + Critério C": ("A1", "in_c"),
    "M2 A1 sem exclusão": ("A1", "aorta_ok"),
    "M3 A2 sem exclusão": ("A2", None),
    "M4 A6 + Critério C": ("A6", "in_c"),
    "M5 A6 sem exclusão": ("A6", None),
}


def sweep_task(args):
    method, p = args
    df = CASES_G
    anc, mask = METHODS[method]
    sub = df if mask is None else df[df[mask]]
    a = anchor(sub, anc)
    o_lo, o_hi = offsets(sub, a, p)
    rv, loss = evaluate(sub.img_id.to_numpy(), arr(sub, "s"), a + o_lo, a + o_hi)
    return dict(metodo=method, p=p, n=len(sub), caixa_y=o_hi[0] - o_lo[0], caixa_x=o_hi[1] - o_lo[1],
                caixa_z=o_hi[2] - o_lo[2], **summarize(rv, loss))


def add_label_bounds(df):
    df[[f"lmin_{a}" for a in AX]] = np.array([RAW[i].min(axis=0) for i in df.img_id], float)
    df[[f"lmax_{a}" for a in AX]] = np.array([RAW[i].max(axis=0) for i in df.img_id], float)
    return df


def worker_init_full(ids):
    global CASES_G
    init_worker(ids)
    CASES_G = add_label_bounds(load_cases())


CASES_G = None

if __name__ == "__main__":
    OUTDIR.mkdir(exist_ok=True)
    cases = load_cases()
    assert len(cases) == 792
    init_worker(cases.img_id)
    cases = add_label_bounds(cases)
    print({m: int(len(cases) if k is None else cases[k].sum()) for m, (_, k) in METHODS.items()})

    # --- Parte A: varredura de p ---
    tasks = [(m, float(p)) for m in METHODS for p in P_GRID]
    with ProcessPoolExecutor(max_workers=6, initializer=worker_init_full, initargs=(list(cases.img_id),)) as ex:
        sweep = pd.DataFrame(list(ex.map(sweep_task, tasks)))
    sweep.to_csv(OUTDIR / "p0d_sweep.csv", index=False)

    # Opção 1 de produção (caixa não recalibrada) na sua população (A1 + Critério C)
    c = cases[cases.in_c].reset_index(drop=True)
    lo, hi = [], []
    for r in c.itertuples():
        b0, b1 = compute_full_resolution_bbox((r.anchor_y, r.anchor_x, r.anchor_z), CAL["option1"]["offset_min"],
                                              CAL["option1"]["offset_max"], CAL["downscale_factors"],
                                              (int(r.sy), int(r.sx), int(r.sz)))
        lo.append(b0)
        hi.append(b1)
    rv, loss = evaluate(c.img_id.to_numpy(), arr(c, "s"), np.array(lo, float), np.array(hi, float))
    prod = dict(metodo="Opção 1 de produção", n=len(c), **summarize(rv, loss))
    pd.DataFrame([prod]).to_csv(OUTDIR / "p0d_option1_production.csv", index=False)
    print("Opção 1 produção:", {k: round(v, 3) if isinstance(v, float) else v for k, v in prod.items()})

    # Restrições
    cons = {"(a) perda máx. <= 1%": sweep.loss_max <= 1.0,
            "(b) <= 2% dos casos com perda > 1%": sweep.frac_gt1 <= 2.0,
            "(c) perda média <= 0,156%": sweep.loss_mean <= 0.156}
    rows = []
    for cname, ok in cons.items():
        for m in METHODS:
            s = sweep[ok & (sweep.metodo == m)]
            if s.empty:
                rows.append(dict(restricao=cname, metodo=m))
                continue
            b = s.loc[s.rv_mean.idxmax()]
            rows.append(dict(restricao=cname, **b.to_dict()))
    best = pd.DataFrame(rows)
    best.to_csv(OUTDIR / "p0d_best_under_constraints.csv", index=False)
    pd.set_option("display.width", 300)
    print(best.round(3).to_string(index=False))

    # --- Parte B: FOV descentralizado (624 casos, caixas do Passo 0c conjunto 1) ---
    s624 = cases[cases.in_624].reset_index(drop=True)
    p0c = pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0c/p0c_fixed_box_summary.csv")
    p0c = p0c[p0c.conjunto.str.startswith("1:")]
    pmap = {"A1": float(p0c.loc[p0c.ancora.str.startswith("A1"), "p"].iloc[0]),
            "A2": float(p0c.loc[p0c.ancora.str.startswith("A2"), "p"].iloc[0]),
            "A6": float(p0c.loc[p0c.ancora.str.startswith("A6"), "p"].iloc[0])}
    offs = {a: offsets(s624, anchor(s624, a), pmap[a]) for a in pmap}
    print("p Passo 0c:", pmap, " tamanhos:", {a: np.round(o[1] - o[0], 1).tolist() for a, o in offs.items()})

    shifts = [("nenhum", None, 0, None)]
    for axis_i, ks in ((2, (10, 20, 40)), (1, (32, 64, 96))):
        for k in ks:
            for side in ("inicio", "fim"):
                shifts.append((f"{AX[axis_i]} -{k} ({side})", axis_i, k, side))
    brows = []
    base = {}
    for name, axis_i, k, side in shifts:
        shape = arr(s624, "s").copy()
        valid = np.ones(len(s624), bool)
        shift_vec = np.zeros(3)
        if axis_i is not None:
            if side == "inicio":
                valid = s624[f"lmin_{AX[axis_i]}"].to_numpy() >= k
                shift_vec[axis_i] = k
            else:
                valid = s624[f"lmax_{AX[axis_i]}"].to_numpy() < shape[:, axis_i] - k
            shape[:, axis_i] -= k
        sub = s624[valid].reset_index(drop=True)
        shp = shape[valid]
        a1 = arr(sub, "anchor_") * FACT - shift_vec
        a2 = shp / 2.0
        anchors_ = {"A1": a1, "A2": a2, "A6": np.column_stack([a2[:, 0], a1[:, 1], a2[:, 2]])}
        for a, anc_pos in anchors_.items():
            o_lo, o_hi = offs[a]
            rv, loss = evaluate(sub.img_id.to_numpy(), shp, anc_pos + o_lo, anc_pos + o_hi,
                                shift=shift_vec.astype(np.int16))
            if name == "nenhum":
                base[a] = dict(zip(s624.img_id, loss))
            b_loss = np.array([base[a][i] for i in sub.img_id])
            brows.append(dict(deslocamento=name, n_valido=len(sub), ancora=a, rv_mean=rv.mean(),
                              loss_mean=loss.mean(), n_gt1=int((loss > 1).sum()), loss_max=loss.max(),
                              base_loss_mean=b_loss.mean(), base_n_gt1=int((b_loss > 1).sum())))
    bdf = pd.DataFrame(brows)
    bdf.to_csv(OUTDIR / "p0d_fov_shift.csv", index=False)
    print(bdf.round(3).to_string(index=False))
    codes = {int(i): "".join(nib.aff2axcodes(nib.load(str(Path("D:/Doutorado/dataset/original") / b /
                                                             f"{i}.label.nii.gz")).affine))
             for i, b in s624[["img_id", "block"]].head(5).itertuples(index=False)}
    print("orientação (aff2axcodes) amostra:", codes)
