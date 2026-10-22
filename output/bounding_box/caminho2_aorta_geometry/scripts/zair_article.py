"""Zair et al. (2022), ROI fiel ao artigo, avaliada como recorte axial no ImageCAS.

Artigo (Seção III-B, eq. 1-4 e Fig. 4a): A1(x1, y1) = ascendente, A2(x2, y2) = descendente, com x
vertical (ântero-posterior) e y horizontal (látero-lateral) na imagem. Centro C(x1, y2) = (AP da
ascendente, LL da descendente). Raio R = Dist(C, A2) = |x2 − x1| (só a componente AP), e o raio
usado é R = |x2 − x1| + 2D, D = diâmetro da ascendente. Máscara circular aplicada a todas as fatias.
Aqui: eixo 1 = AP, eixo 0 = LL. Medidas de zair_slice.csv (uma fatia por caso, sem rótulo).
O rótulo entra só como alvo (perda) e no diagnóstico "fatia acima das coronárias"."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

HERE = Path(__file__).parent
P0 = HERE.parent / "p0_cache"
OUT = REPO / "output/bounding_box/caminho2_aorta_geometry/zair_article"
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())

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
zs = pd.read_csv(HERE / "zair_slice.csv")
df = (so.merge(man, on="img_id", how="left").merge(shp, on="img_id").merge(zs, on="img_id", how="left")
      .merge(ac, on="img_id", how="left"))
df["critC"] = df.in_option1_population.fillna(False).astype(bool)
df["zair_ok"] = df.zair_ok.fillna(False).astype(bool)

# resolução integral: reduzido × 2 nos eixos 0 e 1
sp = (df.spy + df.spx) / 2
df["Cc0"] = 2 * df.desc_c0_red                      # LL da descendente
df["Cc1"] = 2 * df.asc_c1_red                       # AP da ascendente
df["dAP_mm"] = (2 * (df.asc_c1_red - df.desc_c1_red)).abs() * sp
df["D_mm"] = 4 * df.asc_r_red * sp                  # faixa larga, na fatia
df["DP_mm"] = 4 * df.ascP_r_red * sp                # faixa de produção, na fatia
df["Ddesc_mm"] = 4 * df.desc_r_red * sp
df["R0_mm"] = df.dAP_mm                              # eq. 4
df["R1_mm"] = df.dAP_mm + df.D_mm                    # Fig. 4 (c, f)
df["R2_mm"] = df.dAP_mm + 2 * df.D_mm                # raio usado no artigo
df["R2P_mm"] = df.dAP_mm + 2 * df.DP_mm
df["fov_mm"] = df.sy * sp

RAW = {}
for i in df.img_id:
    d = np.load(P0 / f"{i}.npz")
    RAW[i] = np.stack([d[a].astype(np.int16) for a in ("y", "x", "z")], axis=1)
df["label_zmax"] = [RAW[i][:, 2].max() for i in df.img_id]
df["label_in_slice"] = [int((RAW[i][:, 2] == z).sum()) if pd.notna(z) else np.nan
                        for i, z in zip(df.img_id, df.z_sel)]
df["slice_above_label_mm"] = (df.z_sel - df.label_zmax) * df.spz

tv = df[df.split == "trainval"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
GRID = np.indices((512, 512)).reshape(2, -1).T.astype(np.float32)


def metrics(sub, inside_fn, area_fn):
    loss, ra = [], []
    for j, r in enumerate(sub.itertuples()):
        loss.append((1 - inside_fn(sub, j, RAW[r.img_id][:, :2].astype(np.float32)).mean()) * 100)
        ra.append((1 - area_fn(sub, j)) * 100)
    loss, ra = np.array(loss), np.array(ra)
    return dict(n=len(sub), perda_media=loss.mean(), pct_gt1=(loss > 1).mean() * 100,
                pct_gt5=(loss > 5).mean() * 100, pct_gt20=(loss > 20).mean() * 100, perda_max=loss.max(),
                reducao_area_media=ra.mean(), reducao_area_mediana=np.median(ra)), loss


def circle(rcol, k=1.0):
    def inside(sub, j, pts):
        r = sub.iloc[j]
        rad = k * r[rcol] / ((r.spy + r.spx) / 2)
        return (pts[:, 0] - r.Cc0) ** 2 + (pts[:, 1] - r.Cc1) ** 2 <= rad ** 2

    def area(sub, j):
        r = sub.iloc[j]
        g = GRID if (r.sy, r.sx) == (512, 512) else np.indices((int(r.sy), int(r.sx))).reshape(2, -1).T
        return inside(sub, j, g).mean()
    return inside, area


def box_fns(lo, hi):
    def inside(sub, j, pts):
        S = sub.iloc[j][["sy", "sx"]].to_numpy(float)
        a, b = np.clip(np.floor(lo[j]), 0, S), np.clip(np.ceil(hi[j]), 0, S)
        return np.all((pts >= a) & (pts < b), axis=1)

    def area(sub, j):
        S = sub.iloc[j][["sy", "sx"]].to_numpy(float)
        a, b = np.clip(np.floor(lo[j]), 0, S), np.clip(np.ceil(hi[j]), 0, S)
        return (b - a).prod() / S.prod()
    return inside, area


def square(sub, rcol, k=1.0):
    c = sub[["Cc0", "Cc1"]].to_numpy(float)
    h = k * sub[rcol].to_numpy(float)[:, None] / sub[["spy", "spx"]].to_numpy(float)
    return box_fns(c - h, c + h)


def opt1(sub):
    lo, hi = [], []
    for r in sub.itertuples():
        b0, b1 = compute_full_resolution_bbox((r.anchor_y, r.anchor_x, r.anchor_z), CAL["option1"]["offset_min"],
                                              CAL["option1"]["offset_max"], CAL["downscale_factors"],
                                              (int(r.sy), int(r.sx), int(r.sz)))
        lo.append(b0[:2])
        hi.append(b1[:2])
    return box_fns(np.array(lo, float), np.array(hi, float))


a2lo = np.array([RAW[i][:, :2].min(axis=0) for i in tv.img_id], float) - tv[["sy", "sx"]].to_numpy(float) / 2
a2hi = np.array([RAW[i][:, :2].max(axis=0) + 1 for i in tv.img_id], float) - tv[["sy", "sx"]].to_numpy(float) / 2


def a2(sub, p):
    c = sub[["sy", "sx"]].to_numpy(float) / 2
    return box_fns(c + np.percentile(a2lo, 100 - p, axis=0), c + np.percentile(a2hi, p, axis=0))


def bisect(fn, target, a, b, n=30):
    for _ in range(n):
        mid = (a + b) / 2
        if fn(mid) > target:
            a = mid
        else:
            b = mid
    return b


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    S = tv[tv.zair_ok & tv.critC & tv.anchor_y.notna()].reset_index(drop=True)
    Z = tv[tv.zair_ok].reset_index(drop=True)
    TE = te[te.zair_ok].reset_index(drop=True)
    target = metrics(S, *opt1(S))[0]["perda_media"]
    pm = lambda sub, fns: metrics(sub, *fns)[0]["perda_media"]  # noqa: E731
    k_c = bisect(lambda k: pm(S, circle("R2_mm", k)), target, 0.3, 3.0)
    k_cP = bisect(lambda k: pm(S, circle("R2P_mm", k)), target, 0.3, 3.0)
    k_s = bisect(lambda k: pm(S, square(S, "R2_mm", k)), target, 0.3, 3.0)
    p_a2 = bisect(lambda p: pm(S, a2(S, p)), target, 80.0, 100.0)

    rows, per = [], {}
    for sname, sub in (("792: mesmos casos (Zair válido ∩ Critério C)", S), ("792: todos com Zair válido", Z),
                       ("208 teste: Zair válido", TE)):
        V = [("Círculo, R = |ΔAP| (eq. 4)", circle("R0_mm")),
             ("Círculo, R = |ΔAP| + D", circle("R1_mm")),
             ("Círculo, R = |ΔAP| + 2D (artigo)", circle("R2_mm")),
             ("Círculo, R = |ΔAP| + 2D, D de produção", circle("R2P_mm")),
             (f"Círculo recalibrado, k = {k_c:.3f} × (|ΔAP| + 2D)", circle("R2_mm", k_c)),
             (f"Círculo recalibrado, k = {k_cP:.3f} × (|ΔAP| + 2D de produção)", circle("R2P_mm", k_cP)),
             ("Quadrado circunscrito, R = |ΔAP| + 2D", square(sub, "R2_mm")),
             (f"Quadrado recalibrado, k = {k_s:.3f}", square(sub, "R2_mm", k_s)),
             (f"A2 axial (p = {p_a2:.2f})", a2(sub, p_a2))]
        if sub.anchor_y.notna().all():
            V.append(("Opção 1 de produção, só eixos 0 e 1", opt1(sub)))
        for vname, fns in V:
            m, loss = metrics(sub, *fns)
            rows.append(dict(conjunto=sname, metodo=vname, **m))
            per[f"{sname} | {vname}"] = pd.Series(loss, index=sub.img_id.to_numpy())
        if 0 < (~sub.critC).sum() < len(sub):
            g = sub[~sub.critC].reset_index(drop=True)
            V2 = [(n, f) for n, f in V if not n.startswith(("Quadrado", "A2", "Opção"))] + \
                 [(f"Quadrado recalibrado, k = {k_s:.3f}", square(g, "R2_mm", k_s)),
                  (f"A2 axial (p = {p_a2:.2f})", a2(g, p_a2)), ("Opção 1 de produção, só eixos 0 e 1", opt1(g))]
            for vname, fns in V2:
                rows.append(dict(conjunto=f"{sname} — só fora de C", metodo=vname, **metrics(g, *fns)[0]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "zair_article_axial.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "zair_article_axial_percase.csv")
    keep = ["img_id", "split", "critC", "zair_ok", "z_sel", "z_top", "asc_wide_found", "Cc0", "Cc1", "dAP_mm",
            "D_mm", "DP_mm", "Ddesc_mm", "R0_mm", "R1_mm", "R2_mm", "R2P_mm", "fov_mm", "label_in_slice",
            "slice_above_label_mm"]
    df[keep].to_csv(OUT / "zair_article_cases.csv", index=False)

    # diagnóstico da fatia e das medidas
    diag = []
    for sname, sub in (("792", Z), ("C dentro", Z[Z.critC]), ("C fora", Z[~Z.critC]), ("208 teste", TE)):
        diag.append(dict(conjunto=sname, n=len(sub), fatia_no_topo=(sub.z_sel == sub.z_top).mean() * 100,
                         fatia_sem_coronaria=(sub.label_in_slice == 0).mean() * 100,
                         fatia_acima_de_todo_label=(sub.slice_above_label_mm > 0).mean() * 100,
                         D_mediana=sub.D_mm.median(), D_p5=sub.D_mm.quantile(.05), D_p95=sub.D_mm.quantile(.95),
                         DP_mediana=sub.DP_mm.median(), Ddesc_mediana=sub.Ddesc_mm.median(),
                         dAP_mediana=sub.dAP_mm.median(), R2_mediana=sub.R2_mm.median(),
                         R2_p5=sub.R2_mm.quantile(.05), R2_p95=sub.R2_mm.quantile(.95),
                         fov_mediana=sub.fov_mm.median(), circulo_maior_que_fov=(2 * sub.R2_mm > sub.fov_mm).mean() * 100))
    diag = pd.DataFrame(diag)
    diag.to_csv(OUT / "zair_article_slice_diagnostics.csv", index=False)
    json.dump(dict(alvo_perda_media=target, k_circulo=k_c, k_circulo_D_producao=k_cP, k_quadrado=k_s, p_A2=p_a2,
                   n_mesmos=len(S), n_zair_792=len(Z), n_zair_teste=len(TE)),
              open(OUT / "zair_article_calibration.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    print(diag.round(2).T.to_string())
    print(json.dumps(dict(alvo=target, k_c=k_c, k_cP=k_cP, k_s=k_s, p=p_a2)))
    print(res.round(3).to_string(index=False))
