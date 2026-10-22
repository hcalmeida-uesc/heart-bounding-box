"""Caminho 2, Passos 2 a 4: alvos do rótulo, relação características × alvos, círculo de Zair.

Só leitura. Depende de c2_features.csv, c2_asc_wide/, ../p0_cache/ (coordenadas do label em resolução
integral) e das tabelas de casos dos passos anteriores. Ajustes só nos 792 fora do teste."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

HERE = Path(__file__).parent
P0 = HERE.parent / "p0_cache"
OUT = REPO / "output/bounding_box/caminho2_aorta_geometry"
AXN = {0: "LL (eixo 0)", 1: "AP (eixo 1)", 2: "CC (eixo 2)"}
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())

# ---------------------------------------------------------------- dados
so = pd.read_csv("D:/Doutorado/Coronary_Segmentation_deep_learning/data_list/split_original.csv")
so = so.rename(columns={"ID": "img_id"})
man = pd.read_csv("D:/Doutorado/dataset/cropped_manifests/manifest_all.csv")[["img_id", "in_option1_population"]]
ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv")
ac = ac[ac.status == "ok"][["img_id", "anchor_y", "anchor_x", "anchor_z"]]
cols = ["img_id", "sy", "sx", "sz", "spy", "spx", "spz"]
shp = pd.concat([pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0/p0_cases.csv")[cols],
                 pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0c/p0c_extra_cases.csv")[cols],
                 pd.read_csv(REPO / "output/bounding_box/a2_center_bbox/a2_test_cases.csv")[cols]]
                ).drop_duplicates("img_id")

fe = pd.read_csv(HERE / "c2_features.csv")
# diâmetro da ascendente com faixa larga (janela da raiz)
dw = []
for i in fe.img_id:
    p = HERE / f"c2_asc_wide/{i}.npy"
    w = np.load(p) if p.exists() else np.zeros((0, 5))
    dw.append(dict(img_id=i, n_root_wide=len(w), asc_rw_red=np.median(w[:, 3]) if len(w) >= 5 else np.nan))
fe = fe.merge(pd.DataFrame(dw), on="img_id")
fe = fe.rename(columns={"asc_D_vox": "ascP_D_vox", "asc_D_mm": "ascP_D_mm", "R_vox": "RP_vox", "R_mm": "RP_mm"})
fe["asc_D_vox"] = 4 * fe.asc_rw_red                      # 2 × raio reduzido × 2 (resolução integral)
fe["asc_D_mm"] = fe.asc_D_vox * (fe.fsp0 + fe.fsp1) / 2
for u in ("vox", "mm"):
    fe[f"R_{u}"] = fe[f"d_{u}"] + 2 * fe[f"asc_D_{u}"]

df = (so.merge(man, on="img_id", how="left").merge(shp, on="img_id").merge(fe, on="img_id", how="left")
      .merge(ac, on="img_id", how="left"))
df["critC"] = df.in_option1_population.fillna(False).astype(bool)
df["asc_found"] = df.asc_found.fillna(False).astype(bool)
df["desc_found"] = df.desc_found.fillna(False).astype(bool)
df["ok"] = df.asc_found & df.desc_found & df.asc_D_vox.notna()
SP = df[["spy", "spx", "spz"]].to_numpy(float)
# spacing axial do cabeçalho (sem rótulo): referência para a escala comum de características e alvos em mm
df["sp_vox"] = df["sp_mm"] = (df.spy + df.spx) / 2

RAW = {}
for i in df.img_id:
    d = np.load(P0 / f"{i}.npz")
    RAW[i] = np.stack([d[a].astype(np.int16) for a in ("y", "x", "z")], axis=1)
lmin = np.array([RAW[i].min(axis=0) for i in df.img_id], float)
lmax = np.array([RAW[i].max(axis=0) for i in df.img_id], float)

# ---------------------------------------------------------------- Passo 2: alvos
asc_c = df[["asc_c0_vox", "asc_c1_vox", "asc_c2_vox"]].to_numpy(float)
desc_c = df[["desc_c0_vox", "desc_c1_vox", "desc_c2_vox"]].to_numpy(float)
a1 = df[["anchor_y", "anchor_x", "anchor_z"]].to_numpy(float) * np.array([2.0, 2.0, 1.0])
a2 = df[["sy", "sx", "sz"]].to_numpy(float) / 2.0
T = {}
for u, s in (("vox", np.ones_like(SP)), ("mm", SP)):
    for k in range(3):
        T[f"ext_{k}_{u}"] = (lmax[:, k] - lmin[:, k] + 1) * s[:, k]
        T[f"ctr_{k}_{u}"] = (lmin[:, k] + lmax[:, k]) / 2 * s[:, k]
        T[f"relasc_min_{k}_{u}"] = (lmin[:, k] - asc_c[:, k]) * s[:, k]
        T[f"relasc_max_{k}_{u}"] = (lmax[:, k] - asc_c[:, k]) * s[:, k]
        T[f"reldesc_min_{k}_{u}"] = (lmin[:, k] - desc_c[:, k]) * s[:, k]
        T[f"reldesc_max_{k}_{u}"] = (lmax[:, k] - desc_c[:, k]) * s[:, k]
        T[f"A1_{k}_{u}"] = a1[:, k] * s[:, k]
        T[f"A2_{k}_{u}"] = a2[:, k] * s[:, k]
for k, v in T.items():
    df[k] = v
TARGETS = [f"{t}_{k}" for t in ("ext", "ctr", "relasc_min", "relasc_max", "reldesc_min", "reldesc_max")
           for k in range(3)]
FEATS = ["sp", "asc_D", "ascP_D", "desc_D", "d", "dLL", "dAP", "R", "asc_c0", "asc_c1", "asc_c2",
         "desc_c0", "desc_c1", "desc_c2"]
# conjuntos de até 3 características por alvo, escolhidos por relevância anatômica
SIZE = {0: ["dLL", "asc_D", "desc_D"], 1: ["dAP", "asc_D", "desc_D"], 2: ["d", "asc_D", "desc_D"]}
POS = {0: ["asc_c0", "desc_c0"], 1: ["asc_c1", "desc_c1"], 2: ["asc_c2", "desc_c2"]}
MULTI = {}
for k in range(3):
    MULTI[f"ext_{k}"] = SIZE[k]
    MULTI[f"ctr_{k}"] = POS[k]
    for t in ("relasc_min", "relasc_max", "reldesc_min", "reldesc_max"):
        MULTI[f"{t}_{k}"] = SIZE[k]

tv = df[df.split == "trainval"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
assert len(tv) == 792 and len(te) == 208
G = tv[tv.ok].reset_index(drop=True)            # população das análises do Passo 3
GROUPS = {"todos": np.ones(len(G), bool), "C dentro": G.critC.to_numpy(), "C fora": ~G.critC.to_numpy()}


def cv_predict(X, y, folds):
    """Previsões fora da amostra pelos 4 folds; também a mediana e a média de treino de cada fold."""
    pred, med, mean = np.empty_like(y), np.empty_like(y), np.empty_like(y)
    for f in np.unique(folds):
        tr, ts = folds != f, folds == f
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[ts] = np.column_stack([np.ones(ts.sum()), X[ts]]) @ beta
        med[ts], mean[ts] = np.median(y[tr]), y[tr].mean()
    return pred, med, mean


def r2_in(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum(), res


def fit_report(X, y, folds, mask):
    pred, med, mean = cv_predict(X, y, folds)
    res = y - pred
    r2o = 1 - (res[mask] ** 2).sum() / ((y[mask] - mean[mask]) ** 2).sum()
    r2i, _ = r2_in(X[mask], y[mask])
    p95r = np.percentile(np.abs(res[mask]), 95)
    p95m = np.percentile(np.abs(y[mask] - med[mask]), 95)
    return dict(r2_in=r2i, r2_oos=r2o, fator_sqrt=np.sqrt(max(0.0, 1 - r2o)),
                p95_resid=p95r, p95_sem_modelo=p95m, razao_p95=p95r / p95m, n=int(mask.sum()))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    folds = G.k_fold.to_numpy(int)

    # ------------------------------------------------ Passo 1: taxa de detecção
    det = []
    for name, sub in (("792 fora do teste", tv), ("C dentro", tv[tv.critC]), ("C fora", tv[~tv.critC]),
                      ("208 teste", te)):
        det.append(dict(conjunto=name, n=len(sub), asc=int(sub.asc_found.sum()), desc=int(sub.desc_found.sum()),
                        completos=int(sub.ok.sum()),
                        fatias_desc_mediana=sub.n_desc_slices[sub.desc_found].median(),
                        fatias_desc_p5=sub.n_desc_slices[sub.desc_found].quantile(.05),
                        frac_fatias_desc_mediana=sub.desc_frac_slices[sub.desc_found].median()))
    pd.DataFrame(det).to_csv(OUT / "c2_detection_rate.csv", index=False)
    keep = ["img_id", "split", "k_fold", "critC", "asc_found", "desc_found", "ok", "n_asc", "n_asc_real",
            "z_root", "n_root_win", "n_root_wide", "n_desc_slices", "desc_support", "desc_frac_slices",
            "desc_c_iqr_mm", "desc_dHU_med", "desc_edge_mm"] + \
           [f"{f}_{u}" for f in FEATS + ["RP"] for u in ("vox", "mm")] + \
           [c for c in df.columns if c.startswith(("ext_", "ctr_", "relasc_", "reldesc_", "A1_", "A2_"))]
    df[keep].to_csv(OUT / "c2_cases.csv", index=False)

    # ------------------------------------------------ Passo 3.1: correlações univariadas
    corr = []
    for u in ("mm", "vox"):
        for t in TARGETS:
            y = G[f"{t}_{u}"].to_numpy(float)
            for f in FEATS:
                x = G[f"{f}_{u}"].to_numpy(float)
                for gname, m in GROUPS.items():
                    pr = stats.pearsonr(x[m], y[m])
                    sr = stats.spearmanr(x[m], y[m])
                    corr.append(dict(unidade=u, alvo=t, caracteristica=f, grupo=gname, n=int(m.sum()),
                                     pearson=pr.statistic, pearson_p=pr.pvalue, spearman=sr.statistic,
                                     spearman_p=sr.pvalue, r2=pr.statistic ** 2))
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "c2_correlations.csv", index=False)

    # ------------------------------------------------ Passo 3.2-3.3: regressões (univariadas e multivariada)
    reg = []
    for u in ("mm", "vox"):
        for t in TARGETS:
            y = G[f"{t}_{u}"].to_numpy(float)
            models = {f: [f] for f in FEATS}
            models["multi: " + " + ".join(MULTI[t])] = MULTI[t]
            models["controle: sp + " + " + ".join(MULTI[t])] = ["sp"] + MULTI[t]
            for mname, fs in models.items():
                X = G[[f"{f}_{u}" for f in fs]].to_numpy(float)
                for gname, m in GROUPS.items():
                    r = fit_report(X, y, folds, m)
                    reg.append(dict(unidade=u, alvo=t, modelo=mname, multivariado=mname.startswith("multi"),
                                    grupo=gname, **r))
    reg = pd.DataFrame(reg)
    reg.to_csv(OUT / "c2_regressions.csv", index=False)
    best = (reg[(reg.grupo == "todos") & ~reg.modelo.str.startswith("controle")].sort_values("r2_oos", ascending=False).groupby(["unidade", "alvo"])
            .head(1).sort_values(["unidade", "alvo"]))
    best.to_csv(OUT / "c2_best_per_target.csv", index=False)

    # ------------------------------------------------ Passo 3.4: posição — erro do centro por âncora
    H = G[G.A1_0_mm.notna()].reset_index(drop=True)
    fh = H.k_fold.to_numpy(int)
    pos = []
    for k in range(3):
        y = H[f"ctr_{k}_mm"].to_numpy(float)
        anchors = {"A1 (centróide da aorta, produção)": H[f"A1_{k}_mm"], "centro da ascendente (raiz)":
                   H[f"asc_c{k}_mm"], "centro da descendente": H[f"desc_c{k}_mm"], "A2 (centro do volume)":
                   H[f"A2_{k}_mm"]}
        Xc = np.column_stack([a.to_numpy(float) for a in anchors.values()])
        Xc = Xc[:, Xc.std(axis=0) > 1e-9]
        pred, _, mean = cv_predict(Xc, y, fh)
        for gname, m in (("todos", np.ones(len(H), bool)), ("C dentro", H.critC.to_numpy()),
                         ("C fora", ~H.critC.to_numpy())):
            for aname, a in anchors.items():
                e = y - a.to_numpy(float)
                med = np.empty_like(e)
                for f in np.unique(fh):
                    med[fh == f] = np.median(e[fh != f])
                pos.append(dict(eixo=AXN[k], ancora=aname, grupo=gname, n=int(m.sum()),
                                p95_abs_erro_mm=np.percentile(np.abs(e - med)[m], 95),
                                mediana_abs_erro_mm=np.median(np.abs(e - med)[m]), vies_mediano_mm=np.median(e[m])))
            pos.append(dict(eixo=AXN[k], ancora="(e) combinação linear de A1, ascendente, descendente e A2",
                            grupo=gname, n=int(m.sum()), p95_abs_erro_mm=np.percentile(np.abs(y - pred)[m], 95),
                            mediana_abs_erro_mm=np.median(np.abs(y - pred)[m]),
                            vies_mediano_mm=np.median((y - pred)[m]),
                            r2_oos=1 - ((y - pred)[m] ** 2).sum() / ((y - mean)[m] ** 2).sum()))
    pos = pd.DataFrame(pos)
    pos.to_csv(OUT / "c2_position_anchors.csv", index=False)

    # ------------------------------------------------ Passo 4: círculo de Zair no plano axial
    def axial_metrics(sub, lo, hi):
        S = sub[["sy", "sx"]].to_numpy(float)
        lo, hi = np.clip(np.floor(lo), 0, S), np.clip(np.ceil(hi), 0, S)
        loss = np.array([(1 - np.all((RAW[i][:, :2] >= lo[j]) & (RAW[i][:, :2] < hi[j]), axis=1).mean()) * 100
                         for j, i in enumerate(sub.img_id)])
        ra = (1 - (hi - lo).prod(axis=1) / S.prod(axis=1)) * 100
        return dict(n=len(sub), perda_media=loss.mean(), pct_gt1=(loss > 1).mean() * 100,
                    pct_gt5=(loss > 5).mean() * 100, pct_gt20=(loss > 20).mean() * 100, perda_max=loss.max(),
                    reducao_area_media=ra.mean(), reducao_area_mediana=np.median(ra)), loss

    def zair_box(sub, k, rcol="R_mm"):
        c = (sub[["asc_c0_vox", "asc_c1_vox"]].to_numpy(float)
             + sub[["desc_c0_vox", "desc_c1_vox"]].to_numpy(float)) / 2
        h = k * sub[rcol].to_numpy(float)[:, None] / sub[["spy", "spx"]].to_numpy(float)
        return c - h, c + h

    def opt1_box(sub):
        lo, hi = [], []
        for r in sub.itertuples():
            b0, b1 = compute_full_resolution_bbox((r.anchor_y, r.anchor_x, r.anchor_z), CAL["option1"]["offset_min"],
                                                  CAL["option1"]["offset_max"], CAL["downscale_factors"],
                                                  (int(r.sy), int(r.sx), int(r.sz)))
            lo.append(b0[:2])
            hi.append(b1[:2])
        return np.array(lo, float), np.array(hi, float)

    # offsets do label em relação ao centro do volume, nos 792 (mesma base da calibração A2)
    a2lo = np.array([RAW[i][:, :2].min(axis=0) for i in tv.img_id], float) - tv[["sy", "sx"]].to_numpy(float) / 2
    a2hi = np.array([RAW[i][:, :2].max(axis=0) + 1 for i in tv.img_id], float) - tv[["sy", "sx"]].to_numpy(float) / 2

    def a2_box(sub, p):
        c = sub[["sy", "sx"]].to_numpy(float) / 2
        return c + np.percentile(a2lo, 100 - p, axis=0), c + np.percentile(a2hi, p, axis=0)

    def bisect(fn, target, a, b):
        """fn decrescente no parâmetro; devolve o menor parâmetro com fn <= target (na tolerância)."""
        for _ in range(50):
            mid = (a + b) / 2
            if fn(mid) > target:
                a = mid
            else:
                b = mid
        return b

    Sset = tv[tv.ok & tv.critC & tv.anchor_y.notna()].reset_index(drop=True)   # mesmos casos
    Zall = tv[tv.ok].reset_index(drop=True)
    o1, _ = axial_metrics(Sset, *opt1_box(Sset))
    target = o1["perda_media"]
    kcal = bisect(lambda k: axial_metrics(Sset, *zair_box(Sset, k))[0]["perda_media"], target, 0.2, 2.0)
    kcalP = bisect(lambda k: axial_metrics(Sset, *zair_box(Sset, k, "RP_mm"))[0]["perda_media"], target, 0.2, 2.0)
    pcal = bisect(lambda p: axial_metrics(Sset, *a2_box(Sset, p))[0]["perda_media"], target, 80.0, 100.0)
    TE = te[te.ok].reset_index(drop=True)
    rows, losses = [], {}
    for setname, sub in (("792: mesmos casos (Zair válido ∩ Critério C)", Sset),
                         ("792: todos com Zair válido", Zall), ("208 teste: Zair válido", TE)):
        variants = [("Zair sem ajuste (k = 1, D da ascendente faixa larga)", zair_box(sub, 1.0)),
                    ("Zair sem ajuste (k = 1, D da ascendente faixa de produção)", zair_box(sub, 1.0, "RP_mm")),
                    (f"Zair recalibrado (k = {kcal:.3f}, faixa larga)", zair_box(sub, kcal)),
                    (f"Zair recalibrado (k = {kcalP:.3f}, faixa de produção)", zair_box(sub, kcalP, "RP_mm")),
                    (f"A2 axial (p = {pcal:.2f}, mesma perda média)", a2_box(sub, pcal))]
        if sub.anchor_y.notna().all():
            variants.append(("Opção 1 de produção, só eixos 0 e 1", opt1_box(sub)))
        for vname, (lo, hi) in variants:
            m, loss = axial_metrics(sub, lo, hi)
            rows.append(dict(conjunto=setname, metodo=vname, **m))
            losses[(setname, vname)] = pd.Series(loss, index=sub.img_id.to_numpy())
        for gname, gm in (("C dentro", sub.critC.to_numpy()), ("C fora", ~sub.critC.to_numpy())):
            if 0 < gm.sum() < len(sub):
                for vname, (lo, hi) in variants:
                    m, _ = axial_metrics(sub[gm].reset_index(drop=True), lo[gm], hi[gm])
                    rows.append(dict(conjunto=f"{setname} — {gname}", metodo=vname, **m))
    zt = pd.DataFrame(rows)
    zt.to_csv(OUT / "c2_zair_axial.csv", index=False)
    pd.DataFrame({f"{s} | {v}": l for (s, v), l in losses.items()}).to_csv(OUT / "c2_zair_axial_percase.csv")
    json.dump(dict(alvo_perda_media=target, k=kcal, k_producao=kcalP, p_A2=pcal, n_mesmos=len(Sset),
                   n_zair_792=len(Zall), n_zair_teste=len(TE)), open(OUT / "c2_zair_calibration.json", "w"),
              indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    print(pd.DataFrame(det).to_string(index=False))
    print(best[best.unidade == "mm"][["alvo", "modelo", "r2_in", "r2_oos", "fator_sqrt", "p95_resid",
                                       "p95_sem_modelo", "razao_p95", "n"]].round(3).to_string(index=False))
    print(pos.round(2).to_string(index=False))
    print(json.dumps(dict(alvo=target, k=kcal, kP=kcalP, p=pcal)))
    print(zt.round(3).to_string(index=False))
