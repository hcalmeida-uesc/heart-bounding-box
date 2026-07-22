"""Diagnóstico complementar (2026-07-26, passo 1): `mask_z_max == volume_shape_z - 1` é raro ou
comum na população?

O diagnóstico de 2026-07-25 (`quantify_aorta_mask_leak.py`) encontrou essa igualdade nos 10 casos
inspecionados (5 piores de cobertura + 5 calibradores) e não sabia dizer se isso discrimina os
casos-problema ou é universal. `mask_z_max` (limite superior do bbox 3D da máscara da aorta) NÃO é
um campo salvo em `all_cases.csv` para os 994 casos — só existe para a amostra de 50 casos já
reprocessada no diagnóstico de 2026-07-22 (`diagnose_z_anchor_dispersion.py`,
`candidate_anchor_z.csv`, coluna `z_max_mask`). Reprocessar os 994 casos completos para obter esse
campo em toda a população está fora do escopo (custo proibitivo, ~40-220s/caso); este script usa
a amostra de 50 já disponível, sem reprocessar nada, e é honesto sobre essa limitação de tamanho
amostral (50 de 994, não exaustivo).

NÃO reprocessa nada. NÃO altera nenhum artefato de produção.

Uso:
    uv run python src/eda/check_z_max_edge_prevalence.py

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico complementar (2026-07-26)", para a
interpretação.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"
CANDIDATE_SAMPLE_PATH = DIAGNOSIS_OUTPUT_DIR / "candidate_anchor_z.csv"
MASK_LEAK_SUMMARY_PATH = DIAGNOSIS_OUTPUT_DIR / "mask_leak_quantification_summary.csv"

# Mesmos cortes de quintil usados no diagnóstico de 2026-07-24 (num_circles_quintile_coverage.csv).
QUINTILE_EDGES = [(0, 83, "Q1 (<=83)"), (84, 99, "Q2 (84-99)"), (100, 112, "Q3 (100-112)"),
                   (113, 130, "Q4 (113-130)"), (131, float("inf"), "Q5 (>=131)")]


def assign_quintile(num_circles: float) -> str:
    for low, high, label in QUINTILE_EDGES:
        if low <= num_circles <= high:
            return label
    raise ValueError(f"num_circles={num_circles} fora de todas as faixas")


def main() -> None:
    all_cases = pd.read_csv(PRODUCTION_DIR / "all_cases.csv")

    sample = pd.read_csv(CANDIDATE_SAMPLE_PATH)
    sample = sample.merge(all_cases[["img_id", "volume_shape_z"]], on="img_id", how="left")
    sample["hits_edge"] = sample["z_max_mask"] == (sample["volume_shape_z"] - 1)
    sample["quintil"] = sample["num_circles_detected"].apply(assign_quintile)

    print(f"Amostra (2026-07-22, 50 casos): fração que atinge a borda = "
          f"{sample['hits_edge'].sum()}/{len(sample)} = {sample['hits_edge'].mean():.1%}")

    by_quintile = sample.groupby("quintil")["hits_edge"].agg(["sum", "count", "mean"])
    by_quintile = by_quintile.reindex([q[2] for q in QUINTILE_EDGES])
    print("\n=== Fração que atinge a borda, por quintil de num_circles_detected (amostra n=50) ===")
    print(by_quintile.rename(columns={"sum": "n_hits", "count": "n", "mean": "fracao"}).to_string())

    sample_path = DIAGNOSIS_OUTPUT_DIR / "z_max_edge_prevalence_sample50.csv"
    sample[["img_id", "block", "num_circles_detected", "quintil", "z_max_mask", "volume_shape_z", "hits_edge"]].to_csv(
        sample_path, index=False
    )
    print(f"\nSalvo: {sample_path}")

    # Cruzamento com os 10 casos-problema já reprocessados no diagnostico de 2026-07-25.
    problem_cases = pd.read_csv(MASK_LEAK_SUMMARY_PATH)
    problem_cases["hits_edge"] = problem_cases["mask_z_max"] == (
        problem_cases.merge(all_cases[["img_id", "volume_shape_z"]], on="img_id", how="left")["volume_shape_z"] - 1
    )
    print(
        f"\n10 casos-problema (5 piores cobertura + 5 calibradores): fração que atinge a borda = "
        f"{problem_cases['hits_edge'].sum()}/{len(problem_cases)} = {problem_cases['hits_edge'].mean():.1%}"
    )

    print(
        "\nConclusao preliminar: comparar a fracao da amostra geral (acima) com a fracao dos 10 "
        "casos-problema para julgar se 'colar na borda' discrimina alguma coisa ou e universal."
    )


if __name__ == "__main__":
    main()
