"""Apresentação da proposta: curva redução de volume × casos com perda > 5% e barras no mesmo nível de redução.

Métricas fora da amostra (4 folds, 792 casos) de a7_sweep.csv / a8_sweep.csv; pontos no nível da Opção 1
de tabela_mesma_reducao_792.csv (ap_freeze.py)."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path("D:/Doutorado/heart-bounding-box")
OUT = REPO / "output/bounding_box/apresentacao_proposta"
IMG = REPO / "docs/img/apresentacao_proposta"
IMG.mkdir(parents=True, exist_ok=True)
s7 = pd.read_csv(REPO / "output/bounding_box/a7_composite_anchor/a7_sweep.csv")
s8 = pd.read_csv(REPO / "output/bounding_box/a8_regime_box/a8_sweep.csv")
t1 = pd.read_csv(OUT / "tabela_mesma_reducao_792.csv")
# cores fixas por método (paleta categórica de referência, slots 1–3, em ordem)
O1, A2, PR = t1["Método"].tolist()
COL = {O1: "#2a78d6", A2: "#1baf7a", PR: "#eb6834"}
SHORT = {O1: "Caixa fixa pela aorta\n(Opção 1)", A2: "Caixa pelo centro\nda imagem",
         PR: "Caixa pela aorta e centro\nda imagem, com verificação\n(proposta)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "svg.fonttype": "none"})
comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
NOTE = "Fora da amostra (validação cruzada em 4 partes), 792 casos; Opção 1: os 724 casos que ela cobre."


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(IMG / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# ---- 1. curva
fig, ax = plt.subplots(figsize=(12, 7.5))
curves = {A2: s7[(s7.ancora == "A2") & (s7.avaliacao == "oos")], PR: s8[(s8.variante == "A8c") & (s8.avaliacao == "oos")]}
for name, s in curves.items():
    s = s.sort_values("p")
    ax.plot(s.frac_gt5, s.rv_mean, color=COL[name], lw=2.5, label=SHORT[name].replace("\n", " "))
o, pr = t1.iloc[0], t1.iloc[2]
ax.plot(o["Casos com perda > 5% (%)"], o["Redução de volume (%)"], "o", ms=13, color=COL[O1], mec="white", mew=2,
        label=SHORT[O1].replace("\n", " "), zorder=5)
ax.plot(pr["Casos com perda > 5% (%)"], pr["Redução de volume (%)"], "o", ms=13, color=COL[PR], mec=INK, mew=2,
        zorder=6)
ax.annotate("Ponto escolhido", (pr["Casos com perda > 5% (%)"], pr["Redução de volume (%)"]), xytext=(40, -55),
            textcoords="offset points", fontsize=16, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate("Opção 1", (o["Casos com perda > 5% (%)"], o["Redução de volume (%)"]), xytext=(-10, 20),
            textcoords="offset points", fontsize=16, color=INK, ha="right")
ax.axhline(o["Redução de volume (%)"], color=MUTED, lw=1, ls=":")
ax.set_xlabel("Casos com perda das coronárias acima de 5% (%)")
ax.set_ylabel("Redução de volume (%)")
ax.set_xlim(0, 10)
ax.xaxis.set_major_formatter(comma)
ax.yaxis.set_major_formatter(comma)
style(ax)
ax.legend(frameon=False, loc="lower right", fontsize=15)
ax.set_title("Redução de volume e perda das coronárias", fontsize=20, loc="left")
fig.text(0.01, 0.01, NOTE, fontsize=12, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "curva_reducao_perda")


# ---- 2 e 3. barras no mesmo nível de redução
def bars(col, title, name, nd):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    names = [PR, A2, O1]                     # de cima para baixo: proposta, centro, Opção 1
    vals = [float(t1.loc[t1["Método"] == n, col].iloc[0]) for n in names]
    y = range(len(names))[::-1]
    ax.barh(list(y), vals, color=[COL[n] for n in names], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yy, f"{v:.{nd}f}%".replace(".", ","), va="center", fontsize=18, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([SHORT[n] for n in names], fontsize=15, color=INK)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%".replace(".", ",")))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.text(0, 1.01, "Mesma redução de volume da Opção 1 (cerca de 49,5%)", transform=ax.transAxes, fontsize=14,
            color=MUTED, va="bottom")
    ax.set_title(title + "\n", fontsize=20, loc="left")
    fig.text(0.01, 0.01, NOTE, fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name)


bars("Casos com perda > 5% (%)", "Casos com perda das coronárias acima de 5%", "barras_casos_perda_5", 1)
bars("Perda máxima em um caso (%)", "Perda máxima das coronárias em um caso", "barras_perda_maxima", 1)
print("ok")
