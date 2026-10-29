"""Rv contra cada medida de risco (fora da amostra, 792): A2, A7, A7f, A7-sp e A8*; Opção 1 marcada."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path("D:/Doutorado/heart-bounding-box")
OUT = REPO / "output/bounding_box/a8_regime_box"
a8 = pd.read_csv(OUT / "a8_sweep.csv")
a7 = pd.read_csv(REPO / "output/bounding_box/a7_composite_anchor/a7_sweep.csv").rename(columns={"ancora": "variante"})
sw = pd.concat([a7[a7.variante == "A2"], a8])
o1 = pd.read_csv(REPO / "output/bounding_box/a7_composite_anchor/a7_option1_reference.csv").iloc[0]
COL = {"A2": "#eb6834", "A7": "#eda100", "A7f": "#e87ba4", "A7-sp": "#1baf7a", "A8a": "#4a3aa7",
       "A8b": "#2a78d6", "A8c": "#e34948", "A8b-sp": "#008300"}
RISK = [("loss_mean", "perda média (%)"), ("frac_gt0", "casos com perda > 0 (%)"),
        ("frac_gt1", "casos com perda > 1% (%)"), ("frac_gt5", "casos com perda > 5% (%)"),
        ("frac_gt20", "casos com perda > 20% (%)"), ("loss_max", "perda máxima (%)")]
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
for ax, (c, lab) in zip(axs.flat, RISK):
    for name, col in COL.items():
        s = sw[(sw.variante == name) & (sw.avaliacao == "oos")].sort_values("p")
        ax.plot(s[c], s.rv_mean, color=col, lw=2, label=name, ls="--" if name in ("A2", "A7f") else "-")
    ax.plot(o1[c], o1.rv_mean, marker="*", ms=16, color="#222222", ls="none", label="Opção 1 de produção (724)")
    ax.set_xlabel(lab)
    ax.set_ylabel("Rv média (%)")
    ax.grid(color="#dddddd", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0, 0].set_xlim(0, 2)
axs[0, 0].legend(frameon=False, fontsize=10)
fig.suptitle("Caixa por regime de confiança: Rv × risco, fora da amostra (4 folds, 792 casos fora do teste); "
             "p de 80 a 100", fontsize=13)
fig.tight_layout()
out = REPO / "docs/img/a8_rv_risco.png"
fig.savefig(out, dpi=80)
print(out)
