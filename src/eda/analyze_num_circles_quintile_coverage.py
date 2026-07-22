"""Diagnóstico complementar (2026-07-24): taxa de falha de cobertura em z por quintil de
`num_circles_detected`, agrupamento não-paramétrico (994 casos -> 906 casos val+test).

Continuação dos diagnósticos de 2026-07-22 (amostra de 50 casos, âncora candidata z_min) e
2026-07-23 (correlação linear/rank em toda a população, offset de produção). Aqui, em vez de
correlação (que assume relação monotônica), agrupa os 906 casos de val+test em quintis de
`num_circles_detected` e reporta a taxa de falha de cobertura por faixa — abordagem
intencionalmente não-paramétrica, para não impor forma funcional (U ou outra) antes de ver o
padrão bruto. NÃO ajusta modelo quadrático nem qualquer outro.

NÃO reprocessa nada: usa só coverage_{val,test}_imagecas_splits_train90.csv (que já trazem
`num_circles_detected`, `bbox_{y,x,z}_{min,max}`, `label_{y,x,z}_{min,max}` e `covered_{y,x,z}`
diretamente — sem precisar de join com all_cases.csv). NÃO altera calibration.json nem nenhum
artefato de produção.

Uso:
    uv run python src/eda/analyze_num_circles_quintile_coverage.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-24)", para o racional
completo e a interpretação dos resultados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
COVERAGE_VAL_PATH = PRODUCTION_DIR / "coverage_val_imagecas_splits_train90.csv"
COVERAGE_TEST_PATH = PRODUCTION_DIR / "coverage_test_imagecas_splits_train90.csv"
ALL_CASES_PATH = PRODUCTION_DIR / "all_cases.csv"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"

N_QUINTILES = 5
TOP_N_WORST = 5

# Os 5 img_id já citados em docs/step6v2_aorta_anchor.md (secao "Achado bloqueante novo") como
# extremos por eixo no split de TREINO (argmin/argmax_case_per_axis do calibration json) - NAO
# sao os piores casos de cobertura em val/test, e servem so de comparacao por coincidencia de
# img_id.
CALIBRATION_EXTREME_CASE_IDS = {934: "y_min (treino)", 428: "y_max/x_min (treino)", 90: "x_max (treino)",
                                 23: "z_min (treino)", 44: "z_max (treino)"}


def load_combined_coverage() -> pd.DataFrame:
    val = pd.read_csv(COVERAGE_VAL_PATH)
    val["split"] = "val"
    test = pd.read_csv(COVERAGE_TEST_PATH)
    test["split"] = "test"
    combined = pd.concat([val, test], ignore_index=True)

    # Passo 1: confirmar a chave de juncao - num_circles_detected ja vem direto nas CSVs de
    # cobertura (mesmo campo persistido em all_cases.csv pelo mesmo batch). Verificamos que os
    # valores batem via join por img_id, em vez de assumir.
    all_cases = pd.read_csv(ALL_CASES_PATH)[["img_id", "num_circles_detected"]].rename(
        columns={"num_circles_detected": "num_circles_detected_all_cases"}
    )
    checked = combined.merge(all_cases, on="img_id", how="left", validate="one_to_one")
    mismatches = checked[
        checked["num_circles_detected"] != checked["num_circles_detected_all_cases"]
    ]
    if len(mismatches) > 0:
        raise ValueError(
            f"num_circles_detected diverge entre coverage e all_cases.csv em {len(mismatches)} "
            "casos - juncao por img_id nao e confiavel."
        )
    print(
        f"Confirmado: num_circles_detected em coverage_{{val,test}} bate com all_cases.csv via "
        f"join por img_id em todos os {len(checked)} casos (coluna ja vinha correta, sem "
        "necessidade de merge para uso)."
    )
    return combined


def compute_excess_by_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Excesso em voxels (resolucao reduzida) fora do bbox calibrado, por eixo - nao so a flag
    binaria `covered_{axis}`."""
    out = df.copy()
    for axis in ("y", "x", "z"):
        under = (out[f"bbox_{axis}_min"] - out[f"label_{axis}_min"]).clip(lower=0)
        over = (out[f"label_{axis}_max"] - out[f"bbox_{axis}_max"]).clip(lower=0)
        out[f"excess_{axis}"] = pd.concat([under, over], axis=1).max(axis=1)
    return out


def compute_quintile_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quintil"], bin_edges = pd.qcut(
        df["num_circles_detected"], N_QUINTILES, labels=False, retbins=True, duplicates="drop"
    )
    rows = []
    for q in sorted(df["quintil"].unique()):
        group = df[df["quintil"] == q]
        row = {
            "quintil": int(q) + 1,
            "num_circles_min": group["num_circles_detected"].min(),
            "num_circles_max": group["num_circles_detected"].max(),
            "n": len(group),
        }
        for axis in ("y", "x", "z"):
            n_fail = int((~group[f"covered_{axis}"]).sum())
            row[f"n_fail_{axis}"] = n_fail
            row[f"taxa_fail_{axis}_%"] = 100.0 * n_fail / len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    DIAGNOSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_combined_coverage()
    print(f"Total val+test: {len(combined)} casos")

    combined = compute_excess_by_axis(combined)

    # --- Passos 2-3: quintis de num_circles_detected x taxa/contagem de falha por eixo ---
    quintile_summary = compute_quintile_summary(combined)
    quintile_path = DIAGNOSIS_OUTPUT_DIR / "num_circles_quintile_coverage.csv"
    quintile_summary.to_csv(quintile_path, index=False)
    print("\n=== Falha de cobertura por quintil de num_circles_detected (906 casos) ===")
    print(quintile_summary.round(2).to_string(index=False))
    print(f"Salvo: {quintile_path}")

    total_fail_z = int((~combined["covered_z"]).sum())
    total_fail_joint = int((~combined["covered"]).sum())
    print(
        f"\nTotal falhas em z (binario): {total_fail_z} de {len(combined)} "
        f"({100 * total_fail_z / len(combined):.2f}%)"
    )
    print(
        f"Total falhas conjuntas (qualquer eixo, `covered`==False): {total_fail_joint} de "
        f"{len(combined)} ({100 * total_fail_joint / len(combined):.2f}%) - este e o numero "
        "'44 casos' ja documentado na secao de cobertura val/test; NAO e especifico do eixo z."
    )

    # --- Passo 4: 5 piores casos por excesso em z (nao binario) ---
    worst_z = combined.sort_values("excess_z", ascending=False).head(TOP_N_WORST)
    worst_z_path = DIAGNOSIS_OUTPUT_DIR / "worst_z_excess_cases.csv"
    worst_z[["img_id", "split", "num_circles_detected", "excess_z", "covered_z"]].to_csv(
        worst_z_path, index=False
    )
    print(f"\n=== {TOP_N_WORST} piores casos por excesso em z (voxels, nao binario) ===")
    print(
        worst_z[["img_id", "split", "num_circles_detected", "excess_z", "covered_z"]]
        .round(2)
        .to_string(index=False)
    )
    print(f"Salvo: {worst_z_path}")

    overlap_calibration_extremes = [
        (img_id, CALIBRATION_EXTREME_CASE_IDS[img_id])
        for img_id in worst_z["img_id"]
        if img_id in CALIBRATION_EXTREME_CASE_IDS
    ]
    print(
        f"\nCoincidencia com os 5 img_id extremos de treino citados na secao 'Achado bloqueante "
        f"novo' (934, 428, 90, 23, 44): {overlap_calibration_extremes if overlap_calibration_extremes else 'nenhuma'}"
    )


if __name__ == "__main__":
    main()
