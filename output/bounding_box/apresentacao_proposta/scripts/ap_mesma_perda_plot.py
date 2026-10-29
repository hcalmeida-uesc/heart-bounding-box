"""Apresentação da proposta: dois painéis (perda média; casos com perda > 5%) com o ponto de mesma perda da Opção 1."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path("D:/Doutorado/heart-bounding-box")
OUT = REPO / "output/bounding_box/apresentacao_proposta"
IMG = REPO / "docs/img/apresentacao_proposta"
s7 = pd.read_csv(REPO / "output/bounding_box/a7_composite_anchor/a7_sweep.csv")
s8 = pd.read_csv(REPO / "output/bounding_box/a8_regime_box/a8_sweep.csv")
o1 = pd.read_csv(REPO / "output/bounding_box/a7_composite_anchor/a7_option1_reference.csv").iloc[0]
pts = pd.read_csv(OUT / "mesma_perda_pontos.csv")
pts = pts[(pts.populacao.astype(str) == "792") & (pts.avaliacao == "fora da amostra")]
A2, PR, O1 = ("Caixa pelo centro da imagem", "Caixa pela aorta e centro da imagem, com verificação (proposta)",
              "Caixa fixa pela aorta (Opção 1)")
COL = {O1: "#2a78d6", A2: "#1baf7a", PR: "#eb6834"}          # mesmas cores das figuras anteriores
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "svg.fonttype": "none"})
comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
curves = {A2: s7[(s7.ancora == "A2") & (s7.avaliacao == "oos")].sort_values("p"),
          PR: s8[(s8.variante == "A8c") & (s8.avaliacao == "oos")].sort_values("p")}
PANELS = [("loss_mean", "Perda média das coronárias (%)", "mesma perda média", 4.0),
          ("frac_gt5", "Casos com perda das coronárias acima de 5% (%)", "mesmos casos com perda > 5%", 20.0)]

fig, axs = plt.subplots(1, 2, figsize=(18, 7.8), sharey=True)
for ax, (key, xlabel, ponto, xmax) in zip(axs, PANELS):
    for name, s in curves.items():
        ax.plot(s[key], s.rv_mean, color=COL[name], lw=2.5, label=name)
    ax.plot(o1[key], o1.rv_mean, "o", ms=13, color=COL[O1], mec="white", mew=2, zorder=5, label=O1)
    ax.axvline(o1[key], color=MUTED, lw=1.5, ls="--")
    ax.text(o1[key], 72.5, "  Mesma perda da Opção 1", fontsize=15, color=INK, va="top")
    for name, dy in ((PR, 14), (A2, -26)):
        r = pts[(pts.metodo == name) & (pts.ponto == ponto)].iloc[0]
        ax.plot(r[key], r.rv_mean, "o", ms=12, color=COL[name], mec=INK, mew=1.8, zorder=6)
        ax.annotate(f"{r.rv_mean:.1f}%".replace(".", ","), (r[key], r.rv_mean), xytext=(12, dy),
                    textcoords="offset points", fontsize=17, color=INK, fontweight="bold")
    ax.annotate(f"{o1.rv_mean:.1f}%".replace(".", ","), (o1[key], o1.rv_mean), xytext=(12, -26),
                textcoords="offset points", fontsize=17, color=INK, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(20, 73)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(comma)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axs[0].set_ylabel("Redução de volume (%)")
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=15)
fig.suptitle("Redução de volume com a mesma perda das coronárias da Opção 1", fontsize=21, x=0.01, ha="left")
fig.text(0.99, 0.955, "Fora da amostra, 792 casos; Opção 1: 724 casos", fontsize=13, color=MUTED, ha="right")
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
for ext in ("png", "svg"):
    fig.savefig(IMG / f"mesma_perda.{ext}", dpi=300)
print("ok")
