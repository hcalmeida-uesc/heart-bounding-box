"""Âncora composta — caixa por regime de confiança (A8) e sinais de plausibilidade geométrica.

Continuação de a7_calibrate.py (reaproveita dados, A7, caixa, perda). Variantes, todas com LL = A2:
- A7, A7f: referências (devem reproduzir a seção A7);
- A7-sp: A7 com tamanho pelo spacing (regressão quantílica, como A7f-sp);
- A8a: suspeitos = sinais antigos; regime confiável (A7) e regime de fallback (A2 corrigida pelo
  viés), cada um com sua caixa;
- A8b: A8a com suspeitos = antigos ∪ S1 ∪ S2 ∪ S3;
- A8c: A8a com suspeitos = S1 ∪ S2 ∪ S3 ∪ sem A1 ∪ sem descendente (∪ entradas incompletas);
- A8b-sp: A8b com tamanho pelo spacing só no regime confiável.
Sinais novos, sem rótulo; limiares aprendidos no treino, só nos casos não suspeitos pelos sinais antigos:
- S1: |A7 − (A2 + viés)| em AP ou CC acima do p99 do treino (A7 = regressão da A7f);
- S2: d (mm) fora de [p0,5; p99,5] do treino;
- S3: qualquer coordenada de A1, ascendente ou descendente fora de [p0,5; p99,5] do treino.
Só leitura."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import a7_calibrate as m

OUT = m.REPO / "output/bounding_box/a8_regime_box"
VARIANTS = ["A7", "A7f", "A7-sp", "A8a", "A8b", "A8c", "A8b-sp"]
REGIME = {"A8a", "A8b", "A8c", "A8b-sp"}          # caixa por regime
SP_REL = {"A7-sp", "A8b-sp"}                       # tamanho pelo spacing (regime confiável / único)
OLD = m.SUSPECT
NOA1 = np.isnan(m.A1).any(1)
NODESC = ~m.df.desc_found.to_numpy()
DMM = m.df.d_mm.to_numpy(float)
COORDS = {f"{n}_{k}": arr[:, k] for n, arr in (("A1", m.A1), ("asc", m.ASC), ("desc", m.DESC)) for k in range(3)}
REG = m.TV                                          # índices dos 792


def reg_fit(use):
    coef = {}
    for k in (1, 2):
        X = m._design(use, k)
        keep = np.r_[True, X[:, 1:].std(axis=0) > 1e-9]
        b = np.zeros(X.shape[1])
        b[keep] = np.linalg.lstsq(X[:, keep], m.CTR[use, k], rcond=None)[0]
        coef[k] = b
    return coef


def reg_pred(coef, idx):
    a = m.A2[idx].copy()
    for k in (1, 2):
        a[:, k] = m._design(idx, k) @ coef[k]
    return a


def signals(par, idx):
    """Sinais novos (bool) com os limiares de `par`. Valores ausentes não acionam (as faltas são sinais à parte)."""
    with np.errstate(invalid="ignore"):
        pr = reg_pred(par["coefS"], idx)
        disc = np.abs(pr[:, 1:] - (m.A2[idx, 1:] + par["bias"][1:]))
        s1 = np.any(disc > par["thr1"], axis=1)
        s2 = (DMM[idx] < par["d_lo"]) | (DMM[idx] > par["d_hi"])
        s3 = np.zeros(len(idx), bool)
        for c, (lo, hi) in par["coord"].items():
            v = COORDS[c][idx]
            s3 |= (v < lo) | (v > hi)
    return dict(S1=s1, S2=s2, S3=s3)


def suspect(variant, par, idx):
    if variant in ("A7", "A7-sp"):
        return ~m.complete[idx]
    if variant in ("A7f", "A8a"):
        return OLD[idx]
    s = signals(par, idx)
    new = s["S1"] | s["S2"] | s["S3"]
    if variant in ("A8b", "A8b-sp"):
        return OLD[idx] | new
    return new | NOA1[idx] | NODESC[idx] | ~m.complete[idx]            # A8c


def fit(variant, tr):
    par = {"bias": np.r_[0.0, [np.median(m.CTR[tr, k] - m.A2[tr, k]) for k in (1, 2)]]}
    if variant in ("A7", "A7-sp"):
        par["coef"] = reg_fit(tr[m.complete[tr]])
        return par
    base = tr[~OLD[tr]]
    par["coefS"] = reg_fit(base)
    if variant in ("A7f", "A8a"):
        par["coef"] = par["coefS"]
        return par
    disc = np.abs(reg_pred(par["coefS"], base)[:, 1:] - (m.A2[base, 1:] + par["bias"][1:]))
    par["thr1"] = np.percentile(disc, 99, axis=0)
    par["d_lo"], par["d_hi"] = np.percentile(DMM[base], [0.5, 99.5])
    par["coord"] = {c: tuple(np.percentile(v[base], [0.5, 99.5])) for c, v in COORDS.items()}
    par["coef"] = reg_fit(tr[~suspect(variant, par, tr)])
    return par


def apply(variant, par, idx):
    fb = suspect(variant, par, idx)
    a = m.A2[idx].copy()
    with np.errstate(invalid="ignore"):
        pr = reg_pred(par["coef"], idx)
    for k in (1, 2):
        a[:, k] = np.where(fb, m.A2[idx, k] + par["bias"][k], pr[:, k])
    return a, fb


def calib(variant, tr, crossfit):
    par = fit(variant, tr)
    if crossfit:
        anc = np.empty((len(tr), 3))
        fb = np.zeros(len(tr), bool)
        for g in np.unique(m.FOLD[tr]):
            k = m.FOLD[tr] == g
            anc[k], fb[k] = apply(variant, fit(variant, tr[~k]), tr[k])
    else:
        anc, fb = apply(variant, par, tr)
    lo, hi = m.LMIN[tr] - anc, m.LMAX[tr] + 1 - anc
    groups = {"conf": ~fb, "fb": fb} if variant in REGIME else {"all": np.ones(len(tr), bool)}
    return dict(variant=variant, par=par, key=(variant, tuple(tr), crossfit), sp=m.SPAX[tr],
                off={g: (lo[s], hi[s], m.SPAX[tr][s]) for g, s in groups.items()},
                n_train={"conf": int((~fb).sum()), "fb": int(fb.sum())})


QC = {}


def _q(cal, g, side, k, p, y, sp):
    key = (cal["key"], g, side, k, round(p, 4))
    if key not in QC:
        q = (100 - p) / 100 if side == "lo" else p / 100
        if 0 < q < 1:
            ab = m.qreg(sp, y, q)
        else:
            ab = m.qreg(sp, y, 0.995 if q == 1 else 0.005)
            r = y - (ab[0] + ab[1] * sp)
            ab = ab + np.array([r.max() if q == 1 else r.min(), 0.0])
        QC[key] = ab
    return QC[key]


def make_box(cal, p, idx):
    v = cal["variant"]
    a, fb = apply(v, cal["par"], idx)
    olo, ohi = np.empty((len(idx), 3)), np.empty((len(idx), 3))
    for g, (lo, hi, sp) in cal["off"].items():
        sel = np.ones(len(idx), bool) if g == "all" else (fb if g == "fb" else ~fb)
        if not sel.any():
            continue
        if v in SP_REL and g in ("all", "conf"):
            for k in range(3):
                for side, y, out in (("lo", lo[:, k], olo), ("hi", hi[:, k], ohi)):
                    ab = _q(cal, g, side, k, p, y, sp)
                    out[sel, k] = ab[0] + ab[1] * m.SPAX[idx[sel]]
        else:
            olo[sel] = np.percentile(lo, 100 - p, axis=0)
            ohi[sel] = np.percentile(hi, p, axis=0)
    lo = np.clip(np.floor(a + olo), 0, m.S[idx])
    hi = np.clip(np.ceil(a + ohi), 0, m.S[idx])
    return lo, hi, fb


def schemes(v):
    return {"oos": [(calib(v, m.TV[m.FOLD[m.TV] != f], True), m.TV[m.FOLD[m.TV] == f]) for f in (1, 2, 3, 4)],
            "in": [(calib(v, m.TV, False), m.TV)], "test": [(calib(v, m.TV, True), m.TE)]}


def evaluate(parts, p, with_loss=True):
    out = [make_box(c, p, ap) + (ap,) for c, ap in parts]
    lo, hi, fb, idx = (np.concatenate([o[j] for o in out]) for j in range(4))
    return idx, lo, hi, fb, (m.case_loss(idx, lo, hi) if with_loss else None)


def summ(idx, lo, hi, loss, fb):
    r = m.summary(idx, lo, hi, loss)
    for g, s in (("conf", ~fb), ("fb", fb)):
        if s.any():
            sr = m.summary(idx[s], lo[s], hi[s], loss[s])
            r.update({f"{g}_{k}": sr[k] for k in ("n", "rv_mean", "rv_median", "loss_mean", "frac_gt0", "frac_gt1",
                                                  "frac_gt5", "frac_gt20", "loss_max", "box_ll", "box_ap", "box_cc")})
        else:
            r[f"{g}_n"] = 0
    return r


def sweep_worker(v):
    rows = []
    for sch, parts in schemes(v).items():
        for p in m.P_GRID:
            idx, lo, hi, fb, loss = evaluate(parts, p)
            rows.append(dict(variante=v, avaliacao=sch, p=p, **summ(idx, lo, hi, loss, fb)))
    return rows


def bisect(parts, target):
    a, b = 80.0, 100.0
    for _ in range(40):
        mid = (a + b) / 2
        idx, lo, hi, _, _ = evaluate(parts, mid, with_loss=False)
        a, b = (mid, b) if m.case_rv(idx, lo, hi).mean() > target else (a, mid)
    return a


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 80)

    # ---- limiares por fold e contagem de sinais (fora da amostra: cada caso com os limiares do seu fold)
    th, flags = [], []
    for f in (0, 1, 2, 3, 4):
        tr = m.TV if f == 0 else m.TV[m.FOLD[m.TV] != f]
        par = fit("A8b", tr)
        row = dict(fold_fora=("nenhum (792)" if f == 0 else f), n_base=int((~OLD[tr]).sum()),
                   S1_AP=par["thr1"][0], S1_CC=par["thr1"][1], S2_d_lo=par["d_lo"], S2_d_hi=par["d_hi"])
        for c, (lo, hi) in par["coord"].items():
            row[f"S3_{c}_lo"], row[f"S3_{c}_hi"] = lo, hi
        th.append(row)
        ap = m.TV[m.FOLD[m.TV] == f] if f else m.TE
        s = signals(par, ap)
        for j, i in enumerate(ap):
            flags.append(dict(img_id=int(m.IDS[i]), conjunto=("792 (oos)" if f else "208 de teste (ajuste nos 792)"),
                              k_fold=int(m.FOLD[i]), antigos=bool(OLD[i]), S1=bool(s["S1"][j]), S2=bool(s["S2"][j]),
                              S3=bool(s["S3"][j]), sem_A1=bool(NOA1[i]), sem_desc=bool(NODESC[i]),
                              antigos_motivos="; ".join(r for r, mk in m.REASONS.items() if mk[i])))
    th = pd.DataFrame(th)
    th.to_csv(OUT / "a8_thresholds.csv", index=False)
    f4 = th.iloc[1:].drop(columns="fold_fora")
    ths = pd.DataFrame({"792 (nenhum fora)": th.iloc[0].drop("fold_fora"), "media_4_folds": f4.mean(),
                        "min_4_folds": f4.min(), "max_4_folds": f4.max()})
    ths.to_csv(OUT / "a8_thresholds_summary.csv")
    print(ths.round(2).to_string())
    flags = pd.DataFrame(flags)
    flags["novos"] = flags.S1 | flags.S2 | flags.S3
    flags.to_csv(OUT / "a8_signal_flags.csv", index=False)
    cnt = []
    for cj, g in flags.groupby("conjunto"):
        for name, mk in (("S1", g.S1), ("S2", g.S2), ("S3", g.S3), ("S1 ∪ S2 ∪ S3", g.novos),
                         ("antigos", g.antigos), ("S1 sem antigos", g.S1 & ~g.antigos),
                         ("S2 sem antigos", g.S2 & ~g.antigos), ("S3 sem antigos", g.S3 & ~g.antigos),
                         ("novos sem antigos", g.novos & ~g.antigos), ("novos e antigos", g.novos & g.antigos),
                         ("antigos sem novos", g.antigos & ~g.novos),
                         ("A8b: antigos ∪ novos", g.antigos | g.novos),
                         ("A8c: novos ∪ sem A1 ∪ sem desc", g.novos | g.sem_A1 | g.sem_desc)):
            cnt.append(dict(conjunto=cj, sinal=name, casos=int(mk.sum())))
    cnt = pd.DataFrame(cnt)
    cnt.to_csv(OUT / "a8_signal_counts.csv", index=False)
    print(cnt.to_string(index=False))
    print(flags[flags.img_id.isin([132, 537])].to_string(index=False))

    # ---- varredura (paralela por variante; reaproveitada se existir)
    if not (OUT / "a8_sweep.csv").exists():
        with ProcessPoolExecutor(max_workers=len(VARIANTS)) as ex:
            rows = [r for rr in ex.map(sweep_worker, VARIANTS) for r in rr]
        pd.DataFrame(rows).to_csv(OUT / "a8_sweep.csv", index=False)
    sweep = pd.read_csv(OUT / "a8_sweep.csv")

    # ---- checagem: A7 e A7f reproduzem a seção anterior
    old = pd.read_csv(m.OUT / "a7_sweep.csv").rename(columns={"ancora": "variante"})
    keys = ["variante", "avaliacao", "p"]
    cols = ["n", "rv_mean", "rv_median", "loss_mean", "frac_gt0", "frac_gt1", "frac_gt5", "frac_gt20", "loss_max",
            "box_ll", "box_ap", "box_cc"]
    chk = old[old.variante.isin(["A7", "A7f"])].merge(sweep, on=keys, suffixes=("_a7", "_a8"))
    dmax = max(float((chk[f"{c}_a7"] - chk[f"{c}_a8"]).abs().max()) for c in cols)
    print("checagem A7/A7f:", len(chk), "linhas; maior diferença", dmax)
    chk.to_csv(OUT / "a8_check_a7.csv", index=False)
    if len(chk) != 2 * 3 * len(m.P_GRID) or dmax > 1e-9:
        sys.exit("A7/A7f não reproduzem a seção anterior — parar")

    # ---- regime de fallback: casos de treino por fold e p a partir do qual o percentil fica entre os 2 maiores
    reg = []
    for v in VARIANTS:
        for f in (1, 2, 3, 4, 0):
            tr = m.TV if f == 0 else m.TV[m.FOLD[m.TV] != f]
            c = calib(v, tr, True)
            n = c["n_train"]["fb"]
            reg.append(dict(variante=v, fold_fora=("nenhum (792, teste)" if f == 0 else f), n_conf_treino=c["n_train"]["conf"],
                            n_fb_treino=n, p_entre_2_maiores=(100 * (n - 2) / (n - 1) if n > 2 else np.nan)))
    reg = pd.DataFrame(reg)
    reg.to_csv(OUT / "a8_regime_train_sizes.csv", index=False)
    print(reg.round(2).to_string(index=False))

    # ---- destaques
    a7hl = pd.read_csv(m.OUT / "a7_highlights.csv")
    CONS = {"(a) perda máx. ≤ 1%": lambda s: s.loss_max <= 1.0,
            "(b) ≤ 2% com perda > 1%": lambda s: s.frac_gt1 <= 2.0,
            "(c) perda média ≤ 0,156%": lambda s: s.loss_mean <= 0.156}
    hl, LOSS = [], {}
    for v in VARIANTS:
        sc = schemes(v)
        for cname, fn in CONS.items():
            for sch in ("oos", "in"):
                s = sweep[(sweep.variante == v) & (sweep.avaliacao == sch)].sort_values("p")
                ok = fn(s)
                r = s[ok].iloc[0].drop(["variante", "avaliacao"]).to_dict() if ok.any() else dict(p=np.nan)
                hl.append(dict(variante=v, ponto=cname, avaliacao=sch, **r))
                if sch == "oos" and ok.any():
                    idx, lo, hi, fb, loss = evaluate(sc["test"], r["p"])
                    hl.append(dict(variante=v, ponto=cname, avaliacao="test (p do oos)",
                                   **{**summ(idx, lo, hi, loss, fb), "p": r["p"]}))
        for sch in ("oos", "in"):
            p = bisect(sc[sch], m.OPT1_RV)
            idx, lo, hi, fb, loss = evaluate(sc[sch], p)
            hl.append(dict(variante=v, ponto="Rv = Opção 1", avaliacao=sch, p=p, **summ(idx, lo, hi, loss, fb)))
            if sch == "oos":
                LOSS[v] = (idx, loss, fb, p)
                sub = np.isin(idx, m.OPT1_IN)
                hl.append(dict(variante=v, ponto="Rv = Opção 1", avaliacao="oos, só os 724", p=p,
                               **summ(idx[sub], lo[sub], hi[sub], loss[sub], fb[sub])))
                idx, lo, hi, fb, loss = evaluate(sc["test"], p)
                hl.append(dict(variante=v, ponto="Rv = Opção 1", avaliacao="test (p do oos)", p=p,
                               **summ(idx, lo, hi, loss, fb)))
        print("destaques", v, flush=True)
    hl = pd.DataFrame(hl)
    hl.to_csv(OUT / "a8_highlights.csv", index=False)
    show = ["variante", "ponto", "avaliacao", "p", "n", "rv_mean", "loss_mean", "frac_gt1", "frac_gt5", "frac_gt20",
            "loss_max", "conf_n", "fb_n", "conf_box_ll", "conf_box_ap", "conf_box_cc", "fb_box_ll", "fb_box_ap",
            "fb_box_cc", "conf_rv_mean", "fb_rv_mean", "conf_loss_mean", "fb_loss_mean", "conf_loss_max", "fb_loss_max"]
    print(hl[show].round(3).to_string(index=False))

    # ---- 10 maiores perdas da A8b no Rv = Opção 1 (oos), com a perda da A2 no mesmo Rv
    pa2 = float(a7hl[(a7hl.ancora == "A2") & (a7hl.ponto == "Rv = Opção 1") & (a7hl.avaliacao == "oos")].p.iloc[0])
    sc2 = m.scheme_cals("A2")["oos"]
    i2, _, _, _, l2 = m.evaluate(sc2, pa2)
    a2loss = dict(zip(m.IDS[i2], l2))
    idx, loss, fb, p = LOSS["A8b"]
    fl = flags.set_index("img_id")
    top = []
    for j in np.argsort(-loss)[:10]:
        i = int(m.IDS[idx[j]])
        q = fl.loc[i] if not isinstance(fl.loc[i], pd.DataFrame) else fl.loc[i].iloc[0]
        sig = [s for s in ("S1", "S2", "S3") if q[s]] + ([q.antigos_motivos] if q.antigos else [])
        top.append(dict(img_id=i, k_fold=int(m.FOLD[idx[j]]), perda_A8b=loss[j], perda_A2_mesmo_Rv=a2loss[i],
                        regime=("fallback" if fb[j] else "confiável"), sinais="; ".join(sig) or "—"))
    top = pd.DataFrame(top)
    top.to_csv(OUT / "a8_top10_loss_A8b.csv", index=False)
    print("p A8b =", p, "; p A2 =", pa2)
    print(top.round(2).to_string(index=False))
