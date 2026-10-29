"""Apresentação da proposta: redução de volume no ponto de mesma perda das coronárias da Opção 1.

Para a proposta (A8c, regras e regimes da seção A8) e o centro da imagem (A2), bisseção contínua em p até
(i) perda média = Opção 1 e (ii) casos com perda > 5% = Opção 1, na curva fora da amostra (4 folds, 792) e
restrita aos 724 casos que a Opção 1 cobre. Os p da versão 792 vão para o teste (ajuste nos 792). Não toca
em a8c_frozen.json. Saídas em output/bounding_box/apresentacao_proposta/ e docs/img/apresentacao_proposta/."""
import numpy as np
import pandas as pd

import a7_calibrate as m
import a8_calibrate as a8

OUT = m.REPO / "output/bounding_box/apresentacao_proposta"
o1 = pd.read_csv(m.OUT / "a7_option1_reference.csv")
hl8 = pd.read_csv(a8.OUT / "a8_highlights.csv")
TGT = {"i": ("loss_mean", float(o1.loss_mean.iloc[0])), "ii": ("frac_gt5", float(o1.frac_gt5.iloc[0]))}
SP = m.df[["spy", "spx", "spz"]].to_numpy(float)
NAME = {"a2": "Caixa pelo centro da imagem", "a8c": "Caixa pela aorta e centro da imagem, com verificação (proposta)",
        "o1": "Caixa fixa pela aorta (Opção 1)"}
PT = {"i": "mesma perda média", "ii": "mesmos casos com perda > 5%"}
SCH = {"a2": m.scheme_cals("A2"), "a8c": a8.schemes("A8c")}


def ev(meth, sch, p):
    if meth == "a8c":
        idx, lo, hi, fb, loss = a8.evaluate(SCH[meth][sch], p)
    else:
        idx, lo, hi, fb, loss = m.evaluate(SCH[meth][sch], p)
        fb = np.zeros(len(idx), bool)
    return idx, lo, hi, fb, loss


def restrict(r, pop):
    if pop == "792":
        return r
    s = np.isin(r[0], m.OPT1_IN)
    return tuple(x[s] for x in r)


def stats(r):
    idx, lo, hi, fb, loss = r
    s = m.summary(idx, lo, hi, loss)
    mm = (hi - lo) * SP[idx]
    s["caixa_mm"] = " × ".join(f"{v:.0f}" for v in np.median(mm, axis=0))
    for g, sel in (("plausivel", ~fb), ("nao_confiavel", fb)):
        s[f"caixa_mm_{g}"] = " × ".join(f"{v:.0f}" for v in np.median(mm[sel], axis=0)) if sel.any() else ""
        s[f"n_{g}"] = int(sel.sum())
    return s


def bisect(meth, pop, key, target):
    a, b = 80.0, 100.0                       # métrica de perda decresce com p
    assert stats(restrict(ev(meth, "oos", a), pop))[key] > target >= stats(restrict(ev(meth, "oos", b), pop))[key]
    for _ in range(30):
        mid = (a + b) / 2
        a, b = (mid, b) if stats(restrict(ev(meth, "oos", mid), pop))[key] > target else (a, mid)
    return b                                  # menor p com métrica ≤ alvo


def row(meth, ponto, pop, aval, p, s):
    return {"metodo": NAME[meth], "ponto": ponto, "populacao": pop, "avaliacao": aval, "p": p, **s}


rows = []
# checagem: ponto congelado e Opção 1 continuam batendo com os destaques
hr = hl8[(hl8.variante == "A8c") & (hl8.ponto == "Rv = Opção 1") & (hl8.avaliacao == "oos")].iloc[0]
sf = stats(ev("a8c", "oos", float(hr.p)))
for k in ("rv_mean", "loss_mean", "frac_gt1", "frac_gt5", "frac_gt20", "loss_max"):
    assert abs(sf[k] - hr[k]) < 1e-9, k
assert abs(o1.rv_mean.iloc[0] - m.OPT1_RV) < 1e-9
print("checagem ponto congelado e Opção 1: ok")
for pop in ("792", "724"):
    for pt, (key, tgt) in TGT.items():
        for meth in ("a2", "a8c"):
            p = bisect(meth, pop, key, tgt)
            rows.append(row(meth, PT[pt], pop, "fora da amostra", p, stats(restrict(ev(meth, "oos", p), pop))))
            if pop == "792":
                rows.append(row(meth, PT[pt], "208", "teste (ajuste nos 792)", p, stats(ev(meth, "test", p))))
            print(pop, pt, meth, round(p, 4), round(rows[-1]["rv_mean"], 2), flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "mesma_perda_pontos.csv", index=False)


def tab(pop, aval, o1row):
    cols = {"rv_mean": "Redução de volume (%)", "loss_mean": "Perda média das coronárias (%)",
            "frac_gt1": "Casos com perda > 1% (%)", "frac_gt5": "Casos com perda > 5% (%)",
            "frac_gt20": "Casos com perda > 20% (%)", "loss_max": "Perda máxima em um caso (%)", "n": "Casos cobertos"}
    out = [{"Método": NAME["o1"], "Ponto": "—", **{v: o1row[k] for k, v in cols.items()}}]
    sub = res[(res.populacao == pop) & (res.avaliacao == aval)]
    for pt in PT.values():
        for meth in ("a2", "a8c"):
            r = sub[(sub.ponto == pt) & (sub.metodo == NAME[meth])].iloc[0]
            out.append({"Método": NAME[meth], "Ponto": pt.capitalize(), **{v: r[k] for k, v in cols.items()}})
    t = pd.DataFrame(out)
    t["Casos cobertos"] = t["Casos cobertos"].astype(int)
    return t


def fmt(v, c):
    if isinstance(v, (float, np.floating)):
        return f"{v:.{2 if c.startswith('Perda média') else 1}f}".replace(".", ",")
    return str(v)


def to_md(t):
    lines = ["| " + " | ".join(t.columns) + " |",
             "|" + "|".join("---" if c in ("Método", "Ponto") else "---:" for c in t.columns) + "|"]
    lines += ["| " + " | ".join(fmt(r[c], c) for c in t.columns) + " |" for _, r in t.iterrows()]
    return "\n".join(lines)


o1_724 = o1.iloc[0].copy()
for name, t in (("tabela_mesma_perda_792", tab("792", "fora da amostra", o1.iloc[0])),
                ("tabela_mesma_perda_724", tab("724", "fora da amostra", o1_724)),
                ("tabela_mesma_perda_teste_208", tab("208", "teste (ajuste nos 792)", o1.iloc[1]))):
    t.to_csv(OUT / f"{name}.csv", index=False)
    md = to_md(t)
    if "teste" in name:
        md += "\n\nCasos de teste selecionados entre os que todas as versões do recorte processam."
    (OUT / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
pd.set_option("display.width", 250)
print(res[["metodo", "ponto", "populacao", "avaliacao", "p", "rv_mean", "caixa_mm", "caixa_mm_plausivel",
           "caixa_mm_nao_confiavel", "n_plausivel", "n_nao_confiavel"]].to_string(index=False))
