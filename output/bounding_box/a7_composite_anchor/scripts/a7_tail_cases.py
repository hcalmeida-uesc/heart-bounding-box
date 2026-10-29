"""Cauda de perda no ponto Rv = Opção 1 (fora da amostra): casos com perda > 5% por âncora, e Opção 1 de
produção nos 724. Depende de a7_calibrate.py e das saídas a7_highlights.csv."""
import numpy as np
import pandas as pd

import a7_calibrate as m

hl = pd.read_csv(m.OUT / "a7_highlights.csv")
rows = []
for name in m.ANCHORS:
    p = hl[(hl.ancora == name) & (hl.ponto == "Rv = Opção 1") & (hl.avaliacao == "oos")].p.iloc[0]
    idx, lo, hi, fb, loss = m.evaluate(m.scheme_cals(name)["oos"], p)
    for j in np.where(loss > 5)[0]:
        rows.append(dict(ancora=name, img_id=int(m.IDS[idx[j]]), perda=round(loss[j], 1), fallback=bool(fb[j]),
                         critC=bool(m.df.critC[idx[j]])))
lo, hi = [], []
for j in m.OPT1_IN:
    b0, b1 = m.compute_full_resolution_bbox(tuple(m.A1[j] / np.array([2.0, 2.0, 1.0])), m.CAL["option1"]["offset_min"],
                                            m.CAL["option1"]["offset_max"], m.CAL["downscale_factors"],
                                            tuple(int(v) for v in m.S[j]))
    lo.append(b0), hi.append(b1)
loss = m.case_loss(m.OPT1_IN, np.array(lo, float), np.array(hi, float))
for j in np.where(loss > 5)[0]:
    rows.append(dict(ancora="Opção 1", img_id=int(m.IDS[m.OPT1_IN[j]]), perda=round(loss[j], 1), fallback=False,
                     critC=True))
pd.DataFrame(rows).to_csv(m.OUT / "a7_tail_cases.csv", index=False)
print("ok", len(rows))
