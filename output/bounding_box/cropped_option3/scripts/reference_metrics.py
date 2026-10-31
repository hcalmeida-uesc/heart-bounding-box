"""Métricas de referência da análise exploratória com os parâmetros congelados (para a verificação 3).

- "congelado, 792": a caixa congelada (calibração nos 792 com cross-fit interno, a mesma do teste)
  aplicada aos próprios 792 — é o que o dataset gerado deve reproduzir nos 792;
- "congelado, 208": a mesma caixa nos 208 de teste (= tabela de teste da apresentação);
- "dentro da amostra (a8_highlights / a8_sweep)": calibração sem cross-fit (offsets das âncoras
  ajustadas no próprio treino), no p do ponto Rv = Opção 1 dentro da amostra e nos p da grade vizinhos.
Rodar de dentro da pasta dos scripts exploratórios (a7_calibrate.py, a8_calibrate.py, p0_cache/)."""
import numpy as np
import pandas as pd

import a7_calibrate as m
import a8_calibrate as a8

OUT = m.REPO / "output/bounding_box/cropped_option3"
hl = pd.read_csv(a8.OUT / "a8_highlights.csv")
sw = pd.read_csv(a8.OUT / "a8_sweep.csv")
P = float(hl[(hl.variante == "A8c") & (hl.ponto == "Rv = Opção 1") & (hl.avaliacao == "oos")].p.iloc[0])
cal = a8.calib("A8c", m.TV, True)
rows = []
for name, idx in (("congelado, 792", m.TV), ("congelado, 208", m.TE)):
    lo, hi, fb = a8.make_box(cal, P, idx)
    loss = m.case_loss(idx, lo, hi)
    rows.append(dict(referencia=name, p=P, **a8.summ(idx, lo, hi, loss, fb)))
r = hl[(hl.variante == "A8c") & (hl.ponto == "Rv = Opção 1") & (hl.avaliacao.isin(["in", "test (p do oos)"]))]
for _, x in r.iterrows():
    rows.append(dict(referencia=f"a8_highlights: {x.avaliacao}", **x.drop(["variante", "ponto", "avaliacao"]).to_dict()))
s = sw[(sw.variante == "A8c") & (sw.avaliacao == "in") & (sw.p.isin([97.0, 97.5]))]
for _, x in s.iterrows():
    rows.append(dict(referencia=f"a8_sweep: in, p = {x.p}", **x.drop(["variante", "avaliacao"]).to_dict()))
ref = pd.DataFrame(rows)
ref.to_csv(OUT / "referencia_exploratoria.csv", index=False)
pd.set_option("display.width", 250)
print(ref[["referencia", "p", "n", "rv_mean", "loss_mean", "frac_gt1", "frac_gt5", "frac_gt20", "loss_max", "conf_n", "fb_n",
           "conf_rv_mean", "fb_rv_mean"]].round(4).to_string(index=False))
