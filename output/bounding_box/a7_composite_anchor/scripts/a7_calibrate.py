"""Âncora composta por eixo (A7) — bbox de tamanho fixo em voxels.

Compara A1, A2, A6, A7, A7 com fallback (A7f) e A7f com tamanho pelo spacing (A7f-sp), com a regra
de caixa dos Passos 0c/0d. Avaliação fora da amostra pelos 4 folds de split_original.csv nos 792 casos
fora do teste (sem Critério C), dentro da amostra nos mesmos 792 e verificação nos 208 de teste.

Só leitura. Depende de c2_cases.csv (Caminho 2), das tabelas de shape/spacing dos passos anteriores,
de all_cases.csv (A1) e de p0_cache/ (coordenadas do label em resolução integral). Eixos LAS: 0
látero-lateral (LL), 1 ântero-posterior (AP), 2 crânio-caudal (CC)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

REPO = Path("D:/Doutorado/heart-bounding-box")
sys.path.insert(0, str(REPO / "src"))
from utils.segmentation.bbox_crop import compute_full_resolution_bbox  # noqa: E402

HERE = Path(__file__).parent
P0 = HERE / "p0_cache"
OUT = REPO / "output/bounding_box/a7_composite_anchor"
CAL = json.loads(Path("D:/Doutorado/dataset/cropped_manifests/calibration_both_options.json").read_text())
P_GRID = np.round(np.concatenate([np.arange(80.0, 99.51, 0.5), np.arange(99.55, 100.001, 0.05)]), 2)
OPT1_RV = float(pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0d/p0d_option1_production.csv").rv_mean.iloc[0])
ANCHORS = ["A1", "A2", "A6", "A7", "A7f", "A7f-sp"]

# ---------------------------------------------------------------- dados
cols = ["img_id", "sy", "sx", "sz", "spy", "spx", "spz"]
shp = pd.concat([pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0/p0_cases.csv")[cols],
                 pd.read_csv(REPO / "output/bounding_box/dynamic_bbox_step0c/p0c_extra_cases.csv")[cols],
                 pd.read_csv(REPO / "output/bounding_box/a2_center_bbox/a2_test_cases.csv")[cols]]
                ).drop_duplicates("img_id")
ac = pd.read_csv(REPO / "output/bounding_box/aorta_anchor/all_cases.csv")
ac = ac[ac.status == "ok"][["img_id", "anchor_y", "anchor_x", "anchor_z"]]
c2 = pd.read_csv(REPO / "output/bounding_box/caminho2_aorta_geometry/c2_cases.csv")
c2 = c2[["img_id", "split", "k_fold", "critC", "asc_found", "desc_found", "desc_frac_slices", "desc_c_iqr_mm",
         "desc_edge_mm", "desc_D_mm", "d_mm"] + [f"{t}_c{k}_vox" for t in ("asc", "desc") for k in range(3)]]
df = c2.merge(shp, on="img_id").merge(ac, on="img_id", how="left").sort_values("img_id").reset_index(drop=True)
df["critC"] = df.critC.fillna(False).astype(bool)
df["desc_found"] = df.desc_found.fillna(False).astype(bool)
df["asc_found"] = df.asc_found.fillna(False).astype(bool)

S = df[["sy", "sx", "sz"]].to_numpy(float)
SPAX = ((df.spy + df.spx) / 2).to_numpy(float)          # spacing axial do cabeçalho (mm)
A1 = df[["anchor_y", "anchor_x", "anchor_z"]].to_numpy(float) * np.array([2.0, 2.0, 1.0])
A2 = S / 2.0
ASC = df[[f"asc_c{k}_vox" for k in range(3)]].to_numpy(float)
DESC = df[[f"desc_c{k}_vox" for k in range(3)]].to_numpy(float)
RAW = {}
for i in df.img_id:
    d = np.load(P0 / f"{i}.npz")
    RAW[i] = np.stack([d[a].astype(np.int16) for a in ("y", "x", "z")], axis=1)
IDS = df.img_id.to_numpy()
LMIN = np.array([RAW[i].min(axis=0) for i in IDS], float)
LMAX = np.array([RAW[i].max(axis=0) for i in IDS], float)
CTR = (LMIN + LMAX) / 2.0
FOLD = df.k_fold.fillna(0).to_numpy(int)

# sinais de suspeita (sem rótulo)
complete = ~np.isnan(A1).any(1) & ~np.isnan(ASC).any(1) & ~np.isnan(DESC).any(1)
REASONS = {
    "fora do Critério C": ~df.critC.to_numpy(),
    "sem aorta de produção (A1)": np.isnan(A1).any(1),
    "descendente não encontrada": ~df.desc_found.to_numpy(),
    "descendente em < 50% das fatias": df.desc_found.to_numpy() & (df.desc_frac_slices < 0.5).to_numpy(),
    "IQR do centro > 10 mm": (df.desc_c_iqr_mm > 10).to_numpy(),
    "círculo cruzando a borda do FOV": (df.desc_edge_mm < 0).to_numpy(),
    "D_desc fora de 10–30 mm": ((df.desc_D_mm < 10) | (df.desc_D_mm > 30)).to_numpy(),
    "d < 30 mm": (df.d_mm < 30).to_numpy(),
}
SUSPECT = np.logical_or.reduce(list(REASONS.values())) | ~complete

TV = np.where(df.split.to_numpy() == "trainval")[0]
TE = np.where(df.split.to_numpy() == "test")[0]
assert len(TV) == 792 and len(TE) == 208
OPT1_IN = TV[df.critC.to_numpy()[TV] & ~np.isnan(A1[TV]).any(1)]
assert len(OPT1_IN) == 724


# ---------------------------------------------------------------- âncoras
def _design(idx, k):
    return np.column_stack([np.ones(len(idx)), A1[idx, k], ASC[idx, k], DESC[idx, k], A2[idx, k]])


def anchor_fit(name, tr):
    """Parâmetros aprendidos nos casos de treino `tr` (sem rótulo na aplicação)."""
    if name not in ("A7", "A7f", "A7f-sp"):
        return None
    use = tr[complete[tr] & (~SUSPECT[tr] if name != "A7" else True)]
    par = {"coef": {}, "bias": {}}
    for k in (1, 2):
        X = _design(use, k)
        keep = np.r_[True, X[:, 1:].std(axis=0) > 1e-9]          # A2 no eixo AP é constante (256)
        beta = np.zeros(X.shape[1])
        beta[keep] = np.linalg.lstsq(X[:, keep], CTR[use, k], rcond=None)[0]
        par["coef"][k] = beta
        par["bias"][k] = float(np.median(CTR[tr, k] - A2[tr, k]))   # A2 corrigida do viés (fallback)
    return par


def anchor_apply(name, par, idx):
    """Âncora (n × 3) e máscara de casos cobertos; `fb` marca os casos que usaram A2 nos eixos 1 e 2."""
    fb = np.zeros(len(idx), bool)
    if name == "A1":
        return A1[idx].copy(), ~np.isnan(A1[idx]).any(1), fb
    if name == "A2":
        return A2[idx].copy(), np.ones(len(idx), bool), fb
    if name == "A6":
        a = A2[idx].copy()
        ok = ~np.isnan(A1[idx, 1])
        a[ok, 1] = A1[idx[ok], 1]
        return a, np.ones(len(idx), bool), ~ok
    a = A2[idx].copy()
    fb = ~complete[idx] if name == "A7" else SUSPECT[idx]
    for k in (1, 2):
        pred = _design(idx, k) @ par["coef"][k]
        a[:, k] = np.where(fb, A2[idx, k] + par["bias"][k], pred)
    return a, np.ones(len(idx), bool), fb


# ---------------------------------------------------------------- caixa
def qreg(x, y, q):
    """Regressão quantílica linear y ~ a + b·x (programação linear, pinball loss)."""
    n = len(y)
    c = np.r_[0.0, 0.0, np.full(n, q), np.full(n, 1 - q)]
    A = np.column_stack([np.ones(n), x, np.eye(n), -np.eye(n)])
    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)
    r = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    return r.x[:2]


QCACHE = {}


def calib(name, tr, crossfit):
    """Aprende âncora e offsets nos casos de treino. Com crossfit, os offsets do treino vêm de âncoras
    previstas fora da amostra dentro do próprio treino (folds internos)."""
    par = anchor_fit(name, tr)
    if crossfit and par is not None:
        anc = np.full((len(tr), 3), np.nan)
        cov = np.zeros(len(tr), bool)
        for g in np.unique(FOLD[tr]):
            m = FOLD[tr] == g
            a, c, _ = anchor_apply(name, anchor_fit(name, tr[~m]), tr[m])
            anc[m], cov[m] = a, c
    else:
        anc, cov, _ = anchor_apply(name, par, tr)
    t = tr[cov]
    lo = LMIN[t] - anc[cov]
    hi = LMAX[t] + 1 - anc[cov]
    return dict(name=name, par=par, lo=lo, hi=hi, sp=SPAX[t], key=(name, tuple(tr), crossfit))


def box_offsets(cal, p, idx):
    """Offsets inferior e superior por caso (n × 3) no percentil p."""
    if cal["name"] != "A7f-sp":
        plo = np.percentile(cal["lo"], 100 - p, axis=0)
        phi = np.percentile(cal["hi"], p, axis=0)
        return np.broadcast_to(plo, (len(idx), 3)), np.broadcast_to(phi, (len(idx), 3))
    olo = np.empty((len(idx), 3))
    ohi = np.empty((len(idx), 3))
    for k in range(3):
        for side, arr_, q, out in (("lo", cal["lo"], (100 - p) / 100, olo), ("hi", cal["hi"], p / 100, ohi)):
            key = (cal["key"], side, k, round(p, 4))
            if key not in QCACHE:
                y = arr_[:, k]
                if 0 < q < 1:
                    ab = qreg(cal["sp"], y, q)
                else:                                                       # p = 100: envelope
                    ab = qreg(cal["sp"], y, 0.995 if q == 1 else 0.005)
                    res = y - (ab[0] + ab[1] * cal["sp"])
                    ab = ab + np.array([res.max() if q == 1 else res.min(), 0.0])
                QCACHE[key] = ab
            a, b = QCACHE[key]
            out[:, k] = a + b * SPAX[idx]
    return olo, ohi


def make_box(cal, p, idx):
    a, cov, fb = anchor_apply(cal["name"], cal["par"], idx)
    olo, ohi = box_offsets(cal, p, idx)
    lo = np.clip(np.floor(a + olo), 0, S[idx])
    hi = np.clip(np.ceil(a + ohi), 0, S[idx])
    return lo, hi, cov, fb


def case_loss(idx, lo, hi):
    out = np.zeros(len(idx))
    inside = np.all((LMIN[idx] >= lo) & (LMAX[idx] + 1 <= hi), axis=1)
    for j in np.where(~inside)[0]:
        r = RAW[IDS[idx[j]]]
        out[j] = (1.0 - np.all((r >= lo[j]) & (r < hi[j]), axis=1).mean()) * 100
    return out


def case_rv(idx, lo, hi):
    return (1.0 - (hi - lo).prod(axis=1) / S[idx].prod(axis=1)) * 100


def summary(idx, lo, hi, loss):
    rv = case_rv(idx, lo, hi)
    ext = hi - lo
    return dict(n=len(idx), rv_mean=rv.mean(), rv_median=np.median(rv), loss_mean=loss.mean(),
                frac_gt0=(loss > 1e-9).mean() * 100, frac_gt1=(loss > 1).mean() * 100,
                frac_gt5=(loss > 5).mean() * 100, frac_gt20=(loss > 20).mean() * 100, loss_max=loss.max(),
                n_gt1=int((loss > 1).sum()), box_ll=np.median(ext[:, 0]), box_ap=np.median(ext[:, 1]),
                box_cc=np.median(ext[:, 2]))


# ---------------------------------------------------------------- esquemas de avaliação
def scheme_cals(name):
    """Para cada esquema: lista de (calibração, índices de aplicação)."""
    folds = [(calib(name, TV[FOLD[TV] != f], True), TV[FOLD[TV] == f]) for f in (1, 2, 3, 4)]
    return {"oos": folds, "in": [(calib(name, TV, False), TV)], "test": [(calib(name, TV, True), TE)]}


def evaluate(parts, p, with_loss=True):
    idx, lo, hi, cov, fb = [], [], [], [], []
    for cal, ap in parts:
        l, h, c, f = make_box(cal, p, ap)
        idx.append(ap), lo.append(l), hi.append(h), cov.append(c), fb.append(f)
    idx, lo, hi, cov, fb = (np.concatenate(x) for x in (idx, lo, hi, cov, fb))
    idx, lo, hi, fb = idx[cov], lo[cov], hi[cov], fb[cov]
    loss = case_loss(idx, lo, hi) if with_loss else None
    return idx, lo, hi, fb, loss


def mean_rv(parts, p):
    idx, lo, hi, _, _ = evaluate(parts, p, with_loss=False)
    return case_rv(idx, lo, hi).mean()


def bisect_rv(parts, target):
    a, b = 80.0, 100.0
    for _ in range(40):
        m = (a + b) / 2
        if mean_rv(parts, m) > target:
            a = m
        else:
            b = m
    return a


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)

    # ---- sinais de suspeita e fallback
    rs = []
    for label, sel in (("792 fora do teste", TV), ("208 de teste", TE)):
        for r, m in REASONS.items():
            rs.append(dict(conjunto=label, motivo=r, casos=int(m[sel].sum())))
        rs.append(dict(conjunto=label, motivo="A7: entradas incompletas (fallback obrigatório)",
                       casos=int((~complete[sel]).sum())))
        rs.append(dict(conjunto=label, motivo="A7f: suspeito (qualquer motivo) → A2 nos eixos AP e CC",
                       casos=int(SUSPECT[sel].sum())))
        rs.append(dict(conjunto=label, motivo="A6: sem A1 → A2 no eixo AP", casos=int(np.isnan(A1[sel, 1]).sum())))
    rs = pd.DataFrame(rs)
    rs.to_csv(OUT / "a7_fallback_counts.csv", index=False)
    print(rs.to_string(index=False))
    m776 = TV[complete[TV]]
    print("checagem dos sinais nos 776 (doc Caminho 2: 17, 9, 25, 5, 2):",
          [int(REASONS[r][m776].sum()) for r in list(REASONS)[3:]])

    # ---- coeficientes de A7/A7f por fold
    coefs = []
    for name in ("A7", "A7f"):
        for f in (0, 1, 2, 3, 4):
            tr = TV if f == 0 else TV[FOLD[TV] != f]
            par = anchor_fit(name, tr)
            for k in (1, 2):
                b = par["coef"][k]
                coefs.append(dict(ancora=name, fold_fora=("nenhum (792)" if f == 0 else f), eixo=k,
                                  intercepto=b[0], A1=b[1], ascendente=b[2], descendente=b[3], A2=b[4],
                                  vies_fallback=par["bias"][k]))
    coefs = pd.DataFrame(coefs)
    coefs.to_csv(OUT / "a7_coefficients.csv", index=False)
    print(coefs.round(3).to_string(index=False))

    # ---- erro de posição (p95 do |centro − âncora − viés|, viés pelo treino) fora da amostra, voxels
    perr = []
    for name in ANCHORS[:5]:
        res = []
        for f in (1, 2, 3, 4):
            tr, ap = TV[FOLD[TV] != f], TV[FOLD[TV] == f]
            par = anchor_fit(name, tr)
            at, ct, _ = anchor_apply(name, par, tr)
            aa, ca, _ = anchor_apply(name, par, ap)
            bias = np.median(CTR[tr][ct] - at[ct], axis=0)
            res.append(CTR[ap][ca] - aa[ca] - bias)
        e = np.abs(np.concatenate(res))
        perr.append(dict(ancora=name, n=len(e), **{f"p95_{a}": np.percentile(e[:, k], 95)
                                                   for k, a in enumerate(("LL", "AP", "CC"))},
                         **{f"mediana_{a}": np.median(e[:, k]) for k, a in enumerate(("LL", "AP", "CC"))}))
    perr = pd.DataFrame(perr)
    perr.to_csv(OUT / "a7_position_error.csv", index=False)
    print(perr.round(2).to_string(index=False))

    # ---- varredura
    SCH = {name: scheme_cals(name) for name in ANCHORS}
    rows, LOSSES = [], {}
    # a varredura salva é reaproveitada (apagar a7_sweep.csv para refazer)
    for name in ([] if (OUT / "a7_sweep.csv").exists() else ANCHORS):
        for sch, parts in SCH[name].items():
            for p in P_GRID:
                idx, lo, hi, fb, loss = evaluate(parts, p)
                rows.append(dict(ancora=name, avaliacao=sch, p=p, **summary(idx, lo, hi, loss)))
        print("varredura", name, flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "a7_sweep.csv", index=False)
    sweep = pd.read_csv(OUT / "a7_sweep.csv")

    # ---- checagem: A2 dentro da amostra deve reproduzir a calibração A2 (voxels)
    ref = pd.read_csv(REPO / "output/bounding_box/a2_center_bbox/a2_sweep.csv")
    ref = ref[ref.unidade == "vox"][["p", "in_rv_mean", "in_rv_median", "in_loss_mean", "in_frac_gt0",
                                     "in_frac_gt1", "in_frac_gt5", "in_frac_gt20", "in_loss_max"]]
    mine = sweep[(sweep.ancora == "A2") & (sweep.avaliacao == "in")].copy()
    chk = ref.merge(mine, on="p")
    diffs = {c: float((chk[f"in_{c}"] - chk[c]).abs().max()) for c in
             ("rv_mean", "rv_median", "loss_mean", "frac_gt0", "frac_gt1", "frac_gt5", "frac_gt20", "loss_max")}
    chk.to_csv(OUT / "a7_check_a2.csv", index=False)
    print("checagem A2 (", len(chk), "valores de p): maior diferença absoluta", diffs)
    if max(diffs.values()) > 1e-6:
        sys.exit("A2 não reproduz a calibração A2 — parar")

    # ---- Opção 1 de produção
    def opt1_box(idx):
        lo, hi = [], []
        for j in idx:
            b0, b1 = compute_full_resolution_bbox(tuple(A1[j] / np.array([2.0, 2.0, 1.0])), CAL["option1"]["offset_min"],
                                                  CAL["option1"]["offset_max"], CAL["downscale_factors"],
                                                  tuple(int(v) for v in S[j]))
            lo.append(b0), hi.append(b1)
        return np.array(lo, float), np.array(hi, float)
    o1 = []
    for label, idx in (("724 de 792 (Opção 1 cobre)", OPT1_IN), ("208 de teste", TE)):
        lo, hi = opt1_box(idx)
        o1.append(dict(conjunto=label, **summary(idx, lo, hi, case_loss(idx, lo, hi))))
    o1 = pd.DataFrame(o1)
    o1.to_csv(OUT / "a7_option1_reference.csv", index=False)
    print(o1.round(3).to_string(index=False))
    assert abs(o1.rv_mean.iloc[0] - OPT1_RV) < 1e-6

    # ---- pontos de destaque
    CONS = {"(a) perda máx. ≤ 1%": lambda s: s.loss_max <= 1.0,
            "(b) ≤ 2% com perda > 1%": lambda s: s.frac_gt1 <= 2.0,
            "(c) perda média ≤ 0,156%": lambda s: s.loss_mean <= 0.156}
    hl = []
    for name in ANCHORS:
        parts = SCH[name]
        # restrições: menor p da grade que satisfaz, separadamente na curva fora e dentro da amostra;
        # o p escolhido na curva fora da amostra (validação cruzada nos 792) é o que vai para o teste
        for cname, fn in CONS.items():
            for sch in ("oos", "in"):
                s = sweep[(sweep.ancora == name) & (sweep.avaliacao == sch)].sort_values("p")
                ok = fn(s)
                r = s[ok].iloc[0].drop(["ancora", "avaliacao"]).to_dict() if ok.any() else dict(p=np.nan)
                hl.append(dict(ancora=name, ponto=cname, avaliacao=sch, **r))
                if sch == "oos" and ok.any():
                    idx, lo, hi, fb, loss = evaluate(parts["test"], r["p"])
                    hl.append(dict(ancora=name, ponto=cname, avaliacao="test (p do oos)",
                                   **{**summary(idx, lo, hi, loss), "p": r["p"]}))
        # Rv igual ao da Opção 1
        for sch in ("oos", "in"):
            p = bisect_rv(parts[sch], OPT1_RV)
            idx, lo, hi, fb, loss = evaluate(parts[sch], p)
            hl.append(dict(ancora=name, ponto="Rv = Opção 1", avaliacao=sch, p=p, **summary(idx, lo, hi, loss)))
            sub = np.isin(idx, OPT1_IN)
            hl.append(dict(ancora=name, ponto="Rv = Opção 1", avaliacao=f"{sch}, só os 724", p=p,
                           **summary(idx[sub], lo[sub], hi[sub], loss[sub])))
            if sch == "oos":
                LOSSES[name] = (idx, loss, fb, p)
                idx, lo, hi, fb, loss = evaluate(parts["test"], p)
                hl.append(dict(ancora=name, ponto="Rv = Opção 1", avaliacao="test (p do oos)", p=p,
                               **summary(idx, lo, hi, loss)))
    hl = pd.DataFrame(hl)
    hl.to_csv(OUT / "a7_highlights.csv", index=False)
    show = ["ancora", "ponto", "avaliacao", "p", "n", "box_ll", "box_ap", "box_cc", "rv_mean", "rv_median",
            "loss_mean", "frac_gt0", "frac_gt1", "frac_gt5", "frac_gt20", "loss_max", "n_gt1"]
    print(hl[show].round(3).to_string(index=False))

    # ---- 10 maiores perdas de A7f no ponto Rv = Opção 1 (fora da amostra)
    idx, loss, fb, p = LOSSES["A7f"]
    o = np.argsort(-loss)[:10]
    a2_by_id = dict(zip(IDS[LOSSES["A2"][0]], LOSSES["A2"][1]))
    top = pd.DataFrame([dict(img_id=int(IDS[idx[j]]), k_fold=int(FOLD[idx[j]]), perda_A7f=loss[j],
                             perda_A2_mesmo_Rv=a2_by_id[IDS[idx[j]]], fallback=bool(fb[j]),
                             eixos_fallback=("AP e CC" if fb[j] else "nenhum"),
                             motivos="; ".join(r for r, m in REASONS.items() if m[idx[j]]) or "—")
                        for j in o])
    top.to_csv(OUT / "a7_top10_loss_A7f.csv", index=False)
    print("p A7f =", p)
    print(top.round(2).to_string(index=False))
