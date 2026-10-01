"""Passo 0d: gráficos Rv x risco de perda (uma curva por método, Opção 1 de produção marcada)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("D:/Doutorado/heart-bounding-box")
D = REPO / "output/bounding_box/dynamic_bbox_step0d"
sweep = pd.concat([pd.read_csv(D / "p0d_sweep.csv"), pd.read_csv(D / "p0d_sweep_fine_99.5_100.csv")])
sweep = sweep.sort_values(["metodo", "p"])
prod = pd.read_csv(D / "p0d_option1_production.csv").iloc[0]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # paleta categórica de referência, slots 1-5 em ordem fixa + marcador/traço como 2ª codificação
    "M1 A1 + Critério C": ("#2a78d6", "o", "-"),
    "M2 A1 sem exclusão": ("#eb6834", "s", "--"),
    "M3 A2 sem exclusão": ("#1baf7a", "^", "-"),
    "M4 A6 + Critério C": ("#eda100", "D", "-."),
    "M5 A6 sem exclusão": ("#e87ba4", "v", ":"),
}
PANELS = [("loss_mean", "Perda média (%)"), ("frac_gt0", "Casos com perda > 0 (%)"),
          ("frac_gt1", "Casos com perda > 1% (%)"), ("frac_gt5", "Casos com perda > 5% (%)"),
          ("frac_gt20", "Casos com perda > 20% (%)"), ("loss_max", "Perda máxima (%)")]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharey=True, facecolor=SURFACE)
for ax, (col, xlabel) in zip(axes.flat, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for m, (color, marker, ls) in STYLE.items():
        s = sweep[sweep.metodo == m]
        ax.plot(s[col], s.rv_mean, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=4,
                markevery=4, label=m)
    ax.plot(prod[col], prod.rv_mean, marker="*", markersize=15, color=INK, markeredgecolor=SURFACE,
            markeredgewidth=1.2, linestyle="none", label="Opção 1 de produção (724 casos)", zorder=5)
    ax.set_xlabel(xlabel)
for ax in axes[:, 0]:
    ax.set_ylabel("Rv médio (%)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
fig.text(0.5, 0.005, "Cada curva: p de 80 a 100 (passo 0,5, mais passo 0,05 entre 99,5 e 100). "
         "Fração de casos e perda sobre os casos cobertos pelo método (724, 786 ou 792).",
         ha="center", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
out = REPO / "docs/img/dynamic_bbox_step0d_rv_vs_risk.png"
fig.savefig(out, dpi=130, facecolor=SURFACE)
print(out)
