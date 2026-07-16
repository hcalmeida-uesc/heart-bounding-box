"""Gera gráficos identificando os casos extremos que definem a calibração da etapa 6 v2.

Uso:
    uv run python src/eda/plot_aorta_anchor_extremes.py

Lê os resultados já salvos em ``output/bounding_box/aorta_anchor/`` (produzidos por
``src/run_aorta_anchor.py --splits config/imagecas_splits_train90.json``) e gera duas figuras em
``docs/img/``:

1. ``step6v2_offset_histograms.png`` — distribuição dos offsets (label − âncora) por eixo entre
   todos os casos processados, com o limite calibrado (definido só pelos casos de treino) marcado,
   e o caso específico que definiu cada extremo.
2. ``step6v2_worst_coverage_cases.png`` — os piores casos de cobertura em val/test (fora da
   calibração), por quanto excedem o bbox calibrado e em qual eixo.

Ver ``docs/step6v2_aorta_anchor.md``, seção "Achado bloqueante novo", para o contexto.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
DOCS_IMG_DIR = REPO_ROOT / "docs" / "img"
SPLITS_STEM = "imagecas_splits_train90"

# Paleta categórica simples: extremos marcados por cor + rótulo de texto (não só cor), para não
# depender de percepção de cor para diferenciar os eixos.
COLOR_BULK = "#4C72B0"
COLOR_CALIB = "#222222"
COLOR_EXTREME = "#DD8452"
AXIS_COLORS = {"y": "#4C72B0", "x": "#DD8452", "z": "#55A868"}


def plot_offset_histograms(ok: pd.DataFrame, calibration: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, axis_name in zip(axes, ("y", "x", "z")):
        lo = ok[f"offset_min_{axis_name}"]
        hi = ok[f"offset_max_{axis_name}"]
        bins = np.histogram_bin_edges(pd.concat([lo, hi]), bins=40)

        ax.hist(lo, bins=bins, color=COLOR_BULK, alpha=0.45, label="offset mín. (label_min − âncora)")
        ax.hist(hi, bins=bins, color=COLOR_BULK, alpha=0.85, label="offset máx. (label_max − âncora)")

        cal_min = calibration["offset_min_reduced"][axis_name]
        cal_max = calibration["offset_max_reduced"][axis_name]
        ax.axvline(cal_min, color=COLOR_CALIB, linestyle="--", linewidth=1.5)
        ax.axvline(
            cal_max, color=COLOR_CALIB, linestyle="--", linewidth=1.5, label="limite calibrado (treino)"
        )

        argmin_id = calibration["argmin_case_per_axis"][axis_name]
        argmax_id = calibration["argmax_case_per_axis"][axis_name]
        for value, case_id in ((cal_min, argmin_id), (cal_max, argmax_id)):
            ax.annotate(
                f"caso {case_id}",
                xy=(value, ax.get_ylim()[1] * 0.9),
                xytext=(value, ax.get_ylim()[1] * 1.05),
                ha="center",
                fontsize=8,
                color=COLOR_EXTREME,
                arrowprops=dict(arrowstyle="-", color=COLOR_EXTREME, lw=1),
            )

        ax.set_title(f"eixo {axis_name}")
        ax.set_xlabel("offset (voxels, resolução reduzida)")
        if axis_name == "y":
            ax.set_ylabel(f"nº de casos ({len(ok)} processados)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Distribuição dos offsets (label − âncora) por eixo — {len(ok)} casos\n"
        "linha tracejada = limite calibrado (só casos de treino); laranja = caso que definiu o limite",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = DOCS_IMG_DIR / "step6v2_offset_histograms.png"
    fig.savefig(out_path, dpi=130)
    plt.show()
    plt.close(fig)
    print(f"salvo: {out_path}")


def plot_worst_coverage_cases(coverage: pd.DataFrame, calibration: dict, top_n: int = 20) -> pd.DataFrame:
    offset_min = calibration["offset_min_reduced"]
    offset_max = calibration["offset_max_reduced"]

    excess_by_axis = {}
    for axis_name in ("y", "x", "z"):
        bbox_min = coverage[f"anchor_{axis_name}"] + offset_min[axis_name]
        bbox_max = coverage[f"anchor_{axis_name}"] + offset_max[axis_name]
        under = (bbox_min - coverage[f"label_{axis_name}_min"]).clip(lower=0)
        over = (coverage[f"label_{axis_name}_max"] - bbox_max).clip(lower=0)
        excess_by_axis[axis_name] = pd.concat([under, over], axis=1).max(axis=1)

    excess_df = pd.DataFrame(excess_by_axis)
    excess_df["img_id"] = coverage["img_id"]
    excess_df["split"] = coverage["split"]
    excess_df["worst_axis"] = excess_df[["y", "x", "z"]].idxmax(axis=1)
    excess_df["max_excess"] = excess_df[["y", "x", "z"]].max(axis=1)

    worst = excess_df[excess_df["max_excess"] > 0].sort_values("max_excess", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 7))
    bar_colors = [AXIS_COLORS[a] for a in worst["worst_axis"]]
    ax.barh(
        [f"{row.img_id} ({row.split})" for row in worst.itertuples()],
        worst["max_excess"],
        color=bar_colors,
    )
    ax.invert_yaxis()
    ax.set_xlabel("excesso além do bbox calibrado (voxels, resolução reduzida)")
    ax.set_title(f"{top_n} piores casos de cobertura (val+test) — fora da calibração")
    ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in AXIS_COLORS.values()]
    ax.legend(
        handles,
        [f"eixo {axis_name}" for axis_name in AXIS_COLORS],
        title="eixo com maior excesso",
        loc="lower right",
        fontsize=8,
    )

    plt.tight_layout()
    out_path = DOCS_IMG_DIR / "step6v2_worst_coverage_cases.png"
    fig.savefig(out_path, dpi=130)
    plt.show()
    plt.close(fig)
    print(f"salvo: {out_path}")
    return excess_df


def main() -> None:
    DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_cases = pd.read_csv(OUTPUT_DIR / "all_cases.csv")
    ok = all_cases[all_cases["status"] == "ok"].copy()
    for axis_name in ("y", "x", "z"):
        ok[f"offset_min_{axis_name}"] = ok[f"label_{axis_name}_min"] - ok[f"anchor_{axis_name}"]
        ok[f"offset_max_{axis_name}"] = ok[f"label_{axis_name}_max"] - ok[f"anchor_{axis_name}"]

    calibration = json.loads((OUTPUT_DIR / f"calibration_{SPLITS_STEM}.json").read_text(encoding="utf-8"))
    splits = json.loads(
        (REPO_ROOT / "config" / f"{SPLITS_STEM}.json").read_text(encoding="utf-8")
    )["splits"]
    train_ids = {int(img_id) for img_id in splits["train"]}

    plot_offset_histograms(ok, calibration)

    val = pd.read_csv(OUTPUT_DIR / f"coverage_val_{SPLITS_STEM}.csv")
    test = pd.read_csv(OUTPUT_DIR / f"coverage_test_{SPLITS_STEM}.csv")
    val["split"] = "val"
    test["split"] = "test"
    coverage = pd.concat([val, test], ignore_index=True)
    excess_df = plot_worst_coverage_cases(coverage, calibration)

    num_failing = int((excess_df["max_excess"] > 0).sum())
    print()
    print("Resumo:")
    print(f"  casos processados: {len(ok)}")
    print(f"  casos de treino (calibração): {len(ok[ok['img_id'].isin(train_ids)])}")
    print(f"  casos val+test com falha de cobertura: {num_failing} de {len(excess_df)}")


if __name__ == "__main__":
    # Backend não-interativo só quando rodado como script — evita quebrar o backend inline de
    # notebooks que importam este módulo (ver check_aorta_anchor_extremes.ipynb).
    import matplotlib

    matplotlib.use("Agg")
    main()
