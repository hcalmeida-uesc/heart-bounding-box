"""Diagnóstico complementar (2026-07-23): `num_circles_detected` explica o offset de calibração?

Extensão, para a população inteira (994 casos), do achado do diagnóstico de 2026-07-22
(`diagnose_z_anchor_dispersion.py`): numa amostra de 50 casos, `z_min` da máscara da aorta
correlacionava com `num_circles_detected` (r=−0,68), sugerindo que o rastreamento Hough+Level Set
(não a anatomia) domina parte da variabilidade posicional. Este script testa a mesma ideia contra
o offset de **produção** (`label − anchor_z`, âncora = centróide, não o `z_min` candidato de
ontem) e contra os 994 casos processados, não uma amostra de 50 — e cruza com os 20 piores casos
de cobertura já identificados (`docs/img/step6v2_worst_coverage_cases.png`).

NÃO reprocessa nada: usa só `all_cases.csv`, `calibration_imagecas_splits_train90.json` e
`coverage_{val,test}_imagecas_splits_train90.csv`, já salvos. NÃO altera nenhum artefato de
produção (não chama `plot_worst_coverage_cases`, que reescreveria o PNG em `docs/img/`) — reproduz
localmente só o ranking dos 20 piores casos, com a mesma lógica, para não tocar nenhum arquivo de
produção.

Uso:
    uv run python src/eda/analyze_num_circles_offset_correlation.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-23): num_circles_detected
explica o offset de calibração?", para o racional completo e a interpretação dos resultados.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
PRODUCTION_ALL_CASES = PRODUCTION_DIR / "all_cases.csv"
COVERAGE_VAL_PATH = PRODUCTION_DIR / "coverage_val_imagecas_splits_train90.csv"
COVERAGE_TEST_PATH = PRODUCTION_DIR / "coverage_test_imagecas_splits_train90.csv"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"

TOP_N_WORST = 20


def compute_correlations(ok: pd.DataFrame) -> pd.DataFrame:
    """Pearson e Spearman entre offset_{min,max}_z (bruto e desvio absoluto da mediana) e
    `num_circles_detected`, nos 994 casos."""
    rows = []
    for col in ("offset_min_z", "offset_max_z"):
        pearson_r, pearson_p = stats.pearsonr(ok[col], ok["num_circles_detected"])
        spearman_r, spearman_p = stats.spearmanr(ok[col], ok["num_circles_detected"])
        rows.append(
            {
                "comparacao": f"{col} (bruto) x num_circles_detected",
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

    for col, dev_col in (
        ("offset_min_z", "abs_dev_offset_min_z"),
        ("offset_max_z", "abs_dev_offset_max_z"),
    ):
        pearson_r, pearson_p = stats.pearsonr(ok[dev_col], ok["num_circles_detected"])
        spearman_r, spearman_p = stats.spearmanr(ok[dev_col], ok["num_circles_detected"])
        rows.append(
            {
                "comparacao": f"|{col} - mediana| x num_circles_detected",
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

    return pd.DataFrame(rows)


def compute_sign_summary(ok: pd.DataFrame) -> pd.DataFrame:
    """Casos com poucos círculos (<= mediana) tendem a offset positivo ou negativo? Comparação
    direta de médias/medianas por grupo, complementando o sinal da correlação."""
    median_circles = ok["num_circles_detected"].median()
    low = ok[ok["num_circles_detected"] <= median_circles]
    high = ok[ok["num_circles_detected"] > median_circles]

    rows = []
    for label, group in (("poucos_circulos (<= mediana)", low), ("muitos_circulos (> mediana)", high)):
        rows.append(
            {
                "grupo": label,
                "n": len(group),
                "num_circles_mediana_grupo": group["num_circles_detected"].median(),
                "offset_min_z_medio": group["offset_min_z"].mean(),
                "offset_min_z_mediana": group["offset_min_z"].median(),
                "offset_max_z_medio": group["offset_max_z"].mean(),
                "offset_max_z_mediana": group["offset_max_z"].median(),
            }
        )
    return pd.DataFrame(rows)


def compute_worst_coverage_cases(coverage: pd.DataFrame, calibration: dict, top_n: int = TOP_N_WORST) -> pd.DataFrame:
    """Reproduz localmente o ranking de `plot_aorta_anchor_extremes.plot_worst_coverage_cases`
    (mesma lógica de excesso por eixo), sem chamar a função que replota/resalva o PNG de produção.
    """
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
    excess_df["img_id"] = coverage["img_id"].to_numpy()
    excess_df["split"] = coverage["split"].to_numpy()
    excess_df["num_circles_detected"] = coverage["num_circles_detected"].to_numpy()
    excess_df["worst_axis"] = excess_df[["y", "x", "z"]].idxmax(axis=1)
    excess_df["max_excess"] = excess_df[["y", "x", "z"]].max(axis=1)

    return excess_df[excess_df["max_excess"] > 0].sort_values("max_excess", ascending=False).head(top_n)


def main() -> None:
    DIAGNOSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cases = pd.read_csv(PRODUCTION_ALL_CASES)
    ok = all_cases[all_cases["status"] == "ok"].copy()
    print(f"Casos 'ok' em all_cases.csv: {len(ok)}")

    # --- Passo 1: offsets de produção (âncora = centróide, anchor_z) ---
    ok["offset_min_z"] = ok["label_z_min"] - ok["anchor_z"]
    ok["offset_max_z"] = ok["label_z_max"] - ok["anchor_z"]
    ok["abs_dev_offset_min_z"] = (ok["offset_min_z"] - ok["offset_min_z"].median()).abs()
    ok["abs_dev_offset_max_z"] = (ok["offset_max_z"] - ok["offset_max_z"].median()).abs()

    detail_path = DIAGNOSIS_OUTPUT_DIR / "num_circles_offset_detail.csv"
    ok[
        [
            "img_id",
            "block",
            "anchor_z",
            "label_z_min",
            "label_z_max",
            "offset_min_z",
            "offset_max_z",
            "num_circles_detected",
        ]
    ].to_csv(detail_path, index=False)
    print(f"Salvo: {detail_path}")

    # --- Passos 2-3: correlações ---
    correlations = compute_correlations(ok)
    correlations_path = DIAGNOSIS_OUTPUT_DIR / "num_circles_offset_correlations.csv"
    correlations.to_csv(correlations_path, index=False)
    print("\n=== Correlações (994 casos): offset_z x num_circles_detected ===")
    print(correlations.round(4).to_string(index=False))
    print(f"Salvo: {correlations_path}")

    sign_summary = compute_sign_summary(ok)
    sign_path = DIAGNOSIS_OUTPUT_DIR / "num_circles_offset_sign_summary.csv"
    sign_summary.to_csv(sign_path, index=False)
    print("\n=== Sinal: poucos vs. muitos círculos (corte na mediana) ===")
    print(sign_summary.round(2).to_string(index=False))
    print(f"Salvo: {sign_path}")

    # --- Passo 4: cruzamento com os 20 piores casos de cobertura ---
    calibration = json.loads(
        (PRODUCTION_DIR / "calibration_imagecas_splits_train90.json").read_text(encoding="utf-8")
    )
    val = pd.read_csv(COVERAGE_VAL_PATH)
    val["split"] = "val"
    test = pd.read_csv(COVERAGE_TEST_PATH)
    test["split"] = "test"
    coverage = pd.concat([val, test], ignore_index=True)

    worst = compute_worst_coverage_cases(coverage, calibration, top_n=TOP_N_WORST)
    worst_path = DIAGNOSIS_OUTPUT_DIR / "worst_coverage_num_circles.csv"
    worst[["img_id", "split", "worst_axis", "max_excess", "num_circles_detected"]].to_csv(
        worst_path, index=False
    )

    pop_median = ok["num_circles_detected"].median()
    pop_mean = ok["num_circles_detected"].mean()
    pop_std = ok["num_circles_detected"].std()
    worst_median = worst["num_circles_detected"].median()
    worst_mean = worst["num_circles_detected"].mean()
    below_pop_median = int((worst["num_circles_detected"] < pop_median).sum())

    print(f"\n=== {TOP_N_WORST} piores casos de cobertura x num_circles_detected ===")
    print(
        worst[["img_id", "split", "worst_axis", "max_excess", "num_circles_detected"]]
        .round(2)
        .to_string(index=False)
    )
    print(
        f"\nPopulação (994 casos): mediana={pop_median:.1f} média={pop_mean:.1f} std={pop_std:.1f}"
    )
    print(
        f"20 piores: mediana={worst_median:.1f} média={worst_mean:.1f} | "
        f"{below_pop_median}/{TOP_N_WORST} abaixo da mediana da população"
    )
    print(f"Salvo: {worst_path}")


if __name__ == "__main__":
    main()
