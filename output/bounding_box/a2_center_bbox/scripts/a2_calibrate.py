"""Bbox centrado no volume (A2): calibração nos 792 fora do teste (sem Critério C), avaliação
in-sample (792) e out-of-sample (208 de teste), em voxels e em mm. Só leitura; depende de p0_cache/."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "p0_cache"
OUT = REPO / "output/bounding_box/a2_center_bbox"
AX = ("y", "x", "z")
GRID_RL2 = np.array([256.0, 256.0, 128.0])
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())
P_GRID = np.round(np.concatenate([np.arange(90.0, 99.51, 0.5), np.arange(99.55, 100.001, 0.05)]), 2)
OPT1_RV_792 = float(pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0d/p0d_option1_production.csv")
                    .rv_mean.iloc[0])

so = pd.read_csv("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")
cols = ["img_id", "sy", "sx", "sz", "spy", "spx", "spz"]
allc = pd.concat([pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0/p0_cases.csv")[cols],
                  pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0c/p0c_extra_cases.csv")[cols],
                  pd.read_csv(HERE / "a2_test_cases.csv")[cols]]).drop_duplicates("img_id")
ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv")
allc = allc.merge(ac[["img_id", "anchor_y", "anchor_x", "anchor_z"]], on="img_id", how="left")
train = allc[allc.img_id.isin(so.loc[so.split != "test", "ID"])].sort_values("img_id").reset_index(drop=True)
test = allc[allc.img_id.isin(so.loc[so.split == "test", "ID"])].sort_values("img_id").reset_index(drop=True)
assert len(train) == 792 and len(test) == 208

RAW = {}
for i in allc.img_id:
    d = np.load(CACHE / f"{i}.npz")
    RAW[i] = np.stack([d[a].astype(np.int16) for a in AX], axis=1)


def arr(df, prefix):
    return df[[f"{prefix}{a}" for a in AX]].to_numpy(float)


for df in (train, test):
    df[[f"lmin_{a}" for a in AX]] = np.array([RAW[i].min(axis=0) for i in df.img_id], float)
    df[[f"lmax_{a}" for a in AX]] = np.array([RAW[i].max(axis=0) for i in df.img_id], float)


def offsets(df, unit):
    c = arr(df, "s") / 2.0
    lo = arr(df, "lmin_") - c
    hi = arr(df, "lmax_") + 1 - c
    if unit == "mm":
        sp = arr(df, "sp")
        lo, hi = lo * sp, hi * sp
    return lo, hi


TRAIN_OFF = {u: offsets(train, u) for u in ("vox", "mm")}


def box(df, unit, p):
    lo_off, hi_off = TRAIN_OFF[unit]
    plo, phi = np.percentile(lo_off, 100 - p, axis=0), np.percentile(hi_off, p, axis=0)
    c = arr(df, "s") / 2.0
    if unit == "mm":
        sp = arr(df, "sp")
        lo, hi = c + plo / sp, c + phi / sp
    else:
        lo, hi = c + plo, c + phi
    shape = arr(df, "s")
    return np.clip(np.floor(lo), 0, shape), np.clip(np.ceil(hi), 0, shape), phi - plo


def rv_of(df, lo, hi):
    return (1.0 - (hi - lo).prod(axis=1) / arr(df, "s").prod(axis=1)) * 100


def metrics(df, lo, hi):
    rv = rv_of(df, lo, hi)
    loss = np.array([(1.0 - np.all((RAW[i] >= lo[k]) & (RAW[i] < hi[k]), axis=1).mean()) * 100
                     for k, i in enumerate(df.img_id)])
    eff = (hi - lo) * arr(df, "sp") / GRID_RL2
    out = dict(rv_mean=rv.mean(), rv_median=np.median(rv), loss_mean=loss.mean(),
               frac_gt0=(loss > 1e-9).mean() * 100, frac_gt1=(loss > 1).mean() * 100,
               frac_gt5=(loss > 5).mean() * 100, frac_gt20=(loss > 20).mean() * 100, loss_max=loss.max(),
               n_gt1=int((loss > 1).sum()))
    for j, a in enumerate(AX):
        out[f"eff_{a}_median"] = np.median(eff[:, j])
        out[f"eff_{a}_p95"] = np.percentile(eff[:, j], 95)
    ext = hi - lo
    for j, a in enumerate(AX):
        out[f"box_{a}_vox_median"] = np.median(ext[:, j])
    return out, loss


def evaluate(unit, p):
    row = dict(unidade=unit, p=float(p))
    for name, df in (("in", train), ("out", test)):
        lo, hi, nominal = box(df, unit, p)
        m, _ = metrics(df, lo, hi)
        row.update({f"{name}_{k}": v for k, v in m.items()})
    row.update({f"caixa_nominal_{a}": nominal[j] for j, a in enumerate(AX)})
    return row


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    # dispersão dos offsets nos 792
    disp = []
    sp_med = np.median(arr(train, "sp"), axis=0)
    for unit in ("vox", "mm"):
        lo, hi = TRAIN_OFF[unit]
        for side, o in (("inf", lo), ("sup", hi)):
            for j, a in enumerate(AX):
                disp.append(dict(unidade=unit, lado=side, eixo=a, p5=np.percentile(o[:, j], 5),
                                 mediana=np.median(o[:, j]), p95=np.percentile(o[:, j], 95), dp=o[:, j].std(ddof=1),
                                 dp_em_mm=o[:, j].std(ddof=1) * (sp_med[j] if unit == "vox" else 1.0)))
    disp = pd.DataFrame(disp)
    disp.to_csv(OUT / "a2_offset_dispersion.csv", index=False)

    rows = [evaluate(u, p) for u in ("vox", "mm") for p in P_GRID]
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "a2_sweep.csv", index=False)

    # Opção 1 de produção nos 208 de teste
    lo, hi = [], []
    for r in test.itertuples():
        b0, b1 = compute_full_resolution_bbox((r.anchor_y, r.anchor_x, r.anchor_z), CAL["option1"]["offset_min"],
                                              CAL["option1"]["offset_max"], CAL["downscale_factors"],
                                              (int(r.sy), int(r.sx), int(r.sz)))
        lo.append(b0)
        hi.append(b1)
    opt1_test, _ = metrics(test, np.array(lo, float), np.array(hi, float))
    pd.DataFrame([opt1_test]).to_csv(OUT / "a2_option1_test.csv", index=False)

    # pontos de destaque
    hl = []
    for unit in ("vox", "mm"):
        a, b = 90.0, 100.0
        for _ in range(40):
            mid = (a + b) / 2
            lo_, hi_, _ = box(train, unit, mid)
            if rv_of(train, lo_, hi_).mean() > OPT1_RV_792:
                a = mid
            else:
                b = mid
        r = evaluate(unit, a)
        r["ponto"] = f"Rv = Opção 1 ({OPT1_RV_792:.2f}%)"
        hl.append(r)
        s = sweep[sweep.unidade == unit].sort_values("p")
        for name, ok in (("perda máx. <= 1%", s.in_loss_max <= 1.0), ("<= 2% com perda > 1%", s.in_frac_gt1 <= 2.0),
                         ("perda média <= 0,156%", s.in_loss_mean <= 0.156)):
            if ok.any():
                r = s[ok].iloc[0].to_dict()
                r["ponto"] = name
                hl.append(r)
    hl = pd.DataFrame(hl)
    hl.to_csv(OUT / "a2_highlights.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 80)
    print("Opção 1 Rv alvo (724 de 792):", OPT1_RV_792)
    print(disp.round(2).to_string(index=False))
    show = ["ponto", "unidade", "p", "caixa_nominal_y", "caixa_nominal_x", "caixa_nominal_z", "in_rv_mean",
            "in_rv_median", "in_loss_mean", "in_frac_gt0", "in_frac_gt1", "in_frac_gt5", "in_frac_gt20",
            "in_loss_max", "out_rv_mean", "out_rv_median", "out_loss_mean", "out_frac_gt0", "out_frac_gt1",
            "out_frac_gt5", "out_frac_gt20", "out_loss_max", "out_n_gt1"]
    print(hl[show].round(3).T.to_string())
    print(hl[["ponto", "unidade"] + [c for c in hl.columns if c.startswith("in_eff") or c.startswith("out_eff")
                                     or c.startswith("in_box")]].round(3).T.to_string())
    print("Opção 1 teste:", {k: round(v, 3) for k, v in opt1_test.items()})
    print(sweep[["unidade", "p", "caixa_nominal_y", "caixa_nominal_x", "caixa_nominal_z", "in_rv_mean",
                 "in_loss_mean", "in_frac_gt1", "in_loss_max", "out_rv_mean", "out_loss_mean", "out_frac_gt1",
                 "out_loss_max"]].round(3).to_string(index=False))
