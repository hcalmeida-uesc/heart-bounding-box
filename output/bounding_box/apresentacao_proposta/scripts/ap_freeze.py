"""Apresentação da proposta (A8c congelada): parâmetros congelados, tabelas e escolha dos casos dos cortes.

Passo 1: ajuste da A8c nos 792 (mesma calibração do esquema de teste da seção A8: âncora, viés e limiares
nos 792; offsets de treino por cross-fit interno), com o p do ponto Rv = Opção 1 da curva fora da amostra.
Passo 3: tabelas fora da amostra (792) e de teste (208) e regimes. Passo 4: tipo de detecção por caso e
caso de perda mais próxima da mediana do tipo. Só leitura dos artefatos; saídas em apresentacao_proposta/."""
import json
import sys

import numpy as np
import pandas as pd

import a7_calibrate as m
import a8_calibrate as a8

OUT = m.REPO / "output/bounding_box/apresentacao_proposta"
OUT.mkdir(parents=True, exist_ok=True)
hl8 = pd.read_csv(a8.OUT / "a8_highlights.csv")
hl7 = pd.read_csv(m.OUT / "a7_highlights.csv")
o1 = pd.read_csv(m.OUT / "a7_option1_reference.csv")
row = hl8[(hl8.variante == "A8c") & (hl8.ponto == "Rv = Opção 1") & (hl8.avaliacao == "oos")].iloc[0]
P = float(row.p)
pa2 = float(hl7[(hl7.ancora == "A2") & (hl7.ponto == "Rv = Opção 1") & (hl7.avaliacao == "oos")].p.iloc[0])
print("p congelado:", repr(P), "; p centro da imagem:", repr(pa2))
AX = ("latero_lateral", "antero_posterior", "cranio_caudal")
SPMED = np.median(m.df[["spy", "spx", "spz"]].to_numpy(float)[m.TV], axis=0)

# ---------------------------------------------------------------- Passo 1: congelar
cal = a8.calib("A8c", m.TV, True)
par = cal["par"]
reg = {}
for g, name in (("conf", "aorta_detectada_e_plausivel"), ("fb", "aorta_nao_confiavel")):
    lo, hi, _ = cal["off"][g]
    olo, ohi = np.percentile(lo, 100 - P, axis=0), np.percentile(hi, P, axis=0)
    reg[name] = dict(
        casos_treino=int(len(lo)),
        offset_inferior_vox=dict(zip(AX, olo.tolist())), offset_superior_vox=dict(zip(AX, ohi.tolist())),
        tamanho_nominal_vox=dict(zip(AX, (ohi - olo).tolist())),
        tamanho_nominal_mm_spacing_mediano=dict(zip(AX, ((ohi - olo) * SPMED).tolist())))
COEF_NAMES = ["intercepto", "aorta_producao", "ascendente", "descendente", "centro_da_imagem"]
frozen = dict(
    metodo="Caixa pela aorta e centro da imagem, com verificação (variante A8c, seção a8_regime_box)",
    ajuste="792 casos fora do teste (split_original.csv); offsets de treino por cross-fit interno nos 4 folds",
    p=P, p_origem="a8_highlights.csv: A8c, ponto 'Rv = Opção 1', avaliação oos",
    unidades="voxels de resolução integral, eixos LAS (0 látero-lateral, 1 ântero-posterior, 2 crânio-caudal)",
    spacing_mediano_mm_792=dict(zip(AX, SPMED.tolist())),
    entradas=dict(
        centro_da_imagem="shape / 2",
        aorta_producao="anchor_(y,x,z) de aorta_anchor/all_cases.csv × (2, 2, 1); ausente se status != ok",
        ascendente="asc_c0..2_vox de caminho2_aorta_geometry (mediana dos centros na janela de 20 mm da raiz)",
        descendente="desc_c0..2_vox de caminho2_aorta_geometry (ausente se desc_found = False)",
        d_mm="distância axial entre os centros da ascendente e da descendente (mm), d_mm de caminho2"),
    regras=dict(
        nao_confiavel=("qualquer de: sem aorta de produção; descendente não encontrada; entrada ausente; "
                       "S1 (|âncora_regressao_sinal − (centro_da_imagem + vies)| > limiar em AP ou CC); "
                       "S2 (d_mm fora de [d_lo, d_hi]); S3 (alguma das 9 coordenadas fora do intervalo)"),
        ancora_confiavel=("LL = centro da imagem; AP e CC = intercepto + Σ coef · [aorta_producao, ascendente, "
                          "descendente, centro_da_imagem] (coeficientes 'regressao_ancora')"),
        ancora_nao_confiavel="LL = centro da imagem; AP e CC = centro da imagem + vies",
        caixa=("inferior = floor(âncora + offset_inferior), superior = ceil(âncora + offset_superior) "
               "(exclusivo), recortados a [0, shape]; offsets do regime do caso")),
    vies_vox=dict(zip(AX, par["bias"].tolist())),
    regressao_ancora={AX[k]: dict(zip(COEF_NAMES, par["coef"][k].tolist())) for k in (1, 2)},
    regressao_sinal_S1={AX[k]: dict(zip(COEF_NAMES, par["coefS"][k].tolist())) for k in (1, 2)},
    limiares=dict(S1_vox=dict(zip(AX[1:], par["thr1"].tolist())), S2_d_mm=[float(par["d_lo"]), float(par["d_hi"])],
                  S3_coordenadas_vox={c: [float(a), float(b)] for c, (a, b) in par["coord"].items()},
                  origem="percentis nos casos de treino não suspeitos pelos sinais antigos (p99; p0,5 e p99,5)"),
    regimes=reg)
(OUT / "a8c_frozen.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(reg, indent=1, ensure_ascii=False))

# a caixa do JSON reproduz a do esquema de teste da seção A8
lo_t, hi_t, fb_t = a8.make_box(cal, P, m.TE)
assert np.array_equal(fb_t, a8.suspect("A8c", par, m.TE))

# ---------------------------------------------------------------- métricas
sc8 = a8.schemes("A8c")
idx8, lo8, hi8, fb8, l8 = a8.evaluate(sc8["oos"], P)
sc2 = m.scheme_cals("A2")
idx2, lo2, hi2, _, l2 = m.evaluate(sc2["oos"], pa2)
it8, lt8, ht8, ft8, ll8 = a8.evaluate(sc8["test"], P)
it2, lt2, ht2, _, ll2 = m.evaluate(sc2["test"], pa2)
NAME = {"o1": "Caixa fixa pela aorta (Opção 1)", "a2": "Caixa pelo centro da imagem",
        "a8c": "Caixa pela aorta e centro da imagem, com verificação (proposta)"}


def trow(name, s):
    return {"Método": name, "Casos cobertos": int(s["n"]), "Redução de volume (%)": s["rv_mean"],
            "Perda média das coronárias (%)": s["loss_mean"], "Casos com perda > 1% (%)": s["frac_gt1"],
            "Casos com perda > 5% (%)": s["frac_gt5"], "Casos com perda > 20% (%)": s["frac_gt20"],
            "Perda máxima em um caso (%)": s["loss_max"]}


t1 = pd.DataFrame([trow(NAME["o1"], o1.iloc[0]), trow(NAME["a2"], m.summary(idx2, lo2, hi2, l2)),
                   trow(NAME["a8c"], m.summary(idx8, lo8, hi8, l8))])
t2 = pd.DataFrame([trow(NAME["o1"], o1.iloc[1]), trow(NAME["a2"], m.summary(it2, lt2, ht2, ll2)),
                   trow(NAME["a8c"], m.summary(it8, lt8, ht8, ll8))])

# checagens contra a8_highlights / a7_highlights
EXP = {NAME["a8c"]: (0.28, 1.6, 16.6), NAME["a2"]: (0.60, 3.5, 24.5), NAME["o1"]: (1.48, 7.2, 61.7)}
for _, r in t1.iterrows():
    got = (round(r["Perda média das coronárias (%)"], 2), round(r["Casos com perda > 5% (%)"], 1),
           round(r["Perda máxima em um caso (%)"], 1))
    print("checagem", r["Método"], got, EXP[r["Método"]])
    if got != EXP[r["Método"]]:
        sys.exit("tabela não bate com os destaques — parar")
for name, src, ev in ((NAME["a8c"], row, None),):
    for c, k in (("rv_mean", "Redução de volume (%)"), ("loss_mean", "Perda média das coronárias (%)"),
                 ("frac_gt1", "Casos com perda > 1% (%)"), ("frac_gt20", "Casos com perda > 20% (%)")):
        assert abs(float(src[c]) - float(t1.loc[t1.Método == name, k].iloc[0])) < 1e-9, c
tr8 = hl8[(hl8.variante == "A8c") & (hl8.ponto == "Rv = Opção 1") & (hl8.avaliacao == "test (p do oos)")].iloc[0]
assert abs(tr8.loss_max - t2.iloc[2]["Perda máxima em um caso (%)"]) < 1e-9

# regimes (fora da amostra, 792): casos, Rv e > 5%; tamanho = caixa congelada × spacing mediano
t3 = []
for g, s, name in (("conf", ~fb8, "Aorta detectada e plausível"), ("fb", fb8, "Aorta não confiável")):
    sz = reg["aorta_detectada_e_plausivel" if g == "conf" else "aorta_nao_confiavel"]["tamanho_nominal_mm_spacing_mediano"]
    sm = m.summary(idx8[s], lo8[s], hi8[s], l8[s])
    assert abs(sm["rv_mean"] - row[f"{g}_rv_mean"]) < 1e-9 and int(sm["n"]) == int(row[f"{g}_n"])
    t3.append({"Situação": name, "Casos (de 792)": int(sm["n"]),
               "Tamanho da caixa (mm, látero-lateral × ântero-posterior × crânio-caudal)": " × ".join(f"{v:.0f}" for v in sz.values()),
               "Redução de volume (%)": sm["rv_mean"], "Casos com perda > 5% (%)": sm["frac_gt5"]})
t3 = pd.DataFrame(t3)


def fmt(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",") if isinstance(v, float) else str(v)


def to_md(t, nd):
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if c in ("Método", "Situação") else "---:" for c in cols) + "|"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], nd.get(c, 1)) for c in cols) + " |")
    return "\n".join(lines)


ND = {"Perda média das coronárias (%)": 2}
for t, f in ((t1, "tabela_mesma_reducao_792"), (t2, "tabela_teste_208"), (t3, "tabela_regimes")):
    t.to_csv(OUT / f"{f}.csv", index=False)
    md = to_md(t, ND)
    if f == "tabela_teste_208":
        md += "\n\nCasos de teste selecionados entre os que todas as versões do recorte processam."
    (OUT / f"{f}.md").write_text(md + "\n", encoding="utf-8")
    print(md)

# ---------------------------------------------------------------- Passo 4: tipo de detecção e caso
i8 = idx8
noa1, nodesc = a8.NOA1[i8], a8.NODESC[i8]
tipo = np.where(noa1, 4, np.where(nodesc, 3, np.where(fb8, 2, 1)))
assert not (fb8 == (tipo == 1)).any()
TIT = {1: "Aorta detectada e plausível — posição pela aorta e pelo centro da imagem",
       2: "Aorta detectada, mas em posição implausível — posição pelo centro da imagem",
       3: "Aorta descendente não encontrada — posição pelo centro da imagem",
       4: "Aorta não detectada — posição pelo centro da imagem"}
pc = pd.DataFrame(dict(img_id=m.IDS[i8].astype(int), k_fold=m.FOLD[i8], tipo=tipo, regime_nao_confiavel=fb8,
                       perda=l8, **{f"caixa_lo{k}": lo8[:, k] for k in range(3)},
                       **{f"caixa_hi{k}": hi8[:, k] for k in range(3)}))
pc.to_csv(OUT / "casos_792_fora_da_amostra.csv", index=False)
sel = []
for t in (1, 2, 3, 4):
    g = pc[pc.tipo == t]
    med = float(np.median(g.perda))
    c = g.assign(dist=(g.perda - med).abs()).sort_values(["dist", "img_id"]).iloc[0]
    sel.append(dict(tipo=t, titulo=TIT[t], casos_do_tipo=len(g), mediana_perda_tipo=med, img_id=int(c.img_id),
                    perda_caso=float(c.perda), k_fold=int(c.k_fold),
                    **{k: int(c[k]) for k in c.index if k.startswith("caixa_")}))
sel = pd.DataFrame(sel)
sel.to_csv(OUT / "casos_cortes.csv", index=False)
print(sel.to_string(index=False))
