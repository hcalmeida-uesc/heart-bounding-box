"""Diagnóstico exploratório (etapa 6 v2): a âncora atual mistura arco+descendente no eixo z?

Hipótese testada: a âncora de produção (centróide 3D da máscara inteira da aorta, Hough+Level Set)
mistura o arco ascendente com o trecho descendente capturado no campo de visão — cuja extensão
varia por protocolo de aquisição, não por anatomia coronária. Parte da dispersão do offset
(label_min/max − âncora) que motivou o bbox de 480×414×312 (ver docs/step6v2_aorta_anchor.md) pode
ser ruído de captura no eixo z, não variabilidade anatômica real entre pacientes.

Compara, numa amostra estratificada por bloco do dataset, a dispersão do offset usando duas
âncoras derivadas da MESMA máscara da aorta:

1. Âncora atual (produção): centróide 3D da máscara inteira (já salvo em
   ``output/bounding_box/aorta_anchor/all_cases.csv``, não recomputado aqui).
2. Âncora candidata (diagnóstico): mesmo y/x do centróide (controle — não deveria mudar), mas z =
   ``compute_proximal_z_anchor`` (z_min do bbox da máscara, lado proximal/raiz confirmado
   empiricamente em 2026-07-21 — ver docs/step6v2_aorta_anchor.md).

NÃO altera nenhum artefato de produção (``calibration_*.json``, CSVs em
``output/bounding_box/aorta_anchor/``) — só lê ``all_cases.csv`` e escreve em
``output/bounding_box/aorta_anchor_z_diagnosis/`` (pasta nova, dedicada a este diagnóstico). O
único reprocessamento é rodar a segmentação da aorta (``segment_aorta``) na amostra, para extrair
o z_min da máscara — label, âncora atual e demais campos vêm todos de ``all_cases.csv``.

Uso:
    uv run python src/eda/diagnose_z_anchor_dispersion.py [--sample-per-block N] [--seed S]

Ver docs/step6v2_aorta_anchor.md, seção "Diagnóstico exploratório (2026-07-22): a âncora atual
mistura arco+descendente no eixo z?", para o racional completo e a interpretação dos resultados.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Scripts em src/eda/ não têm src/ no sys.path automaticamente (diferente de src/run_aorta_anchor.py,
# que fica um nível acima e ganha isso de graça quando invocado como `uv run python src/...py`).
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from utils.project.config import load_config_json
from utils.segmentation.aorta_anchor import segment_aorta
from utils.segmentation.bounding_box import compute_proximal_z_anchor
from utils.segmentation.pipeline_preprocessing import load_and_preprocess_image

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"
DATASET_ROOT = REPO_ROOT / "dataset"
PRODUCTION_ALL_CASES = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor" / "all_cases.csv"
DIAGNOSIS_OUTPUT_DIR = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor_z_diagnosis"

AXES = ("y", "x", "z")


def stratified_sample(all_cases: pd.DataFrame, per_block: int, seed: int) -> pd.DataFrame:
    """Amostra `per_block` casos com status='ok' de cada bloco do dataset, sem reposição."""
    ok = all_cases[all_cases["status"] == "ok"]
    rng = np.random.default_rng(seed)
    samples = []
    for block, group in ok.groupby("block"):
        n = min(per_block, len(group))
        block_seed = int(rng.integers(0, 2**32 - 1))
        samples.append(group.sample(n=n, random_state=block_seed))
    return (
        pd.concat(samples, ignore_index=True)
        .sort_values(["block", "img_id"])
        .reset_index(drop=True)
    )


def compute_candidate_anchor_z(img_id: int, block: str, config: dict) -> dict:
    """Roda etapas 1-4 + segmentação da aorta para 1 caso e extrai o z_min (âncora candidata).

    Único reprocessamento necessário neste diagnóstico: label, âncora atual e demais campos vêm
    de ``all_cases.csv`` (produção), não são recomputados aqui.
    """
    base_path = DATASET_ROOT / block
    image_data = load_and_preprocess_image(img_id, str(base_path), config)
    lcc_image = image_data["lcc_image"]
    scaled_spacing = image_data["scaled_spacing"]

    aorta_mask, num_circles = segment_aorta(lcc_image, scaled_spacing, config)
    if aorta_mask.sum() == 0:
        return {
            "img_id": img_id,
            "block": block,
            "status": "error",
            "error_message": "aorta não detectada nesta reexecução",
        }

    z_min_candidate = compute_proximal_z_anchor(aorta_mask)
    z_max_mask = float(np.argwhere(aorta_mask)[:, 2].max())

    return {
        "img_id": img_id,
        "block": block,
        "status": "ok",
        "error_message": "",
        "z_min_candidate": z_min_candidate,
        "z_max_mask": z_max_mask,
        "num_circles_detected": num_circles,
    }


def compute_offsets(row: dict, anchor: dict[str, float], prefix: str) -> dict[str, float]:
    """offset_min/max = label_min/max - âncora, por eixo, com colunas nomeadas por `prefix`."""
    offsets = {}
    for axis in AXES:
        offsets[f"{prefix}_offset_min_{axis}"] = row[f"label_{axis}_min"] - anchor[axis]
        offsets[f"{prefix}_offset_max_{axis}"] = row[f"label_{axis}_max"] - anchor[axis]
    return offsets


def summarize_dispersion(detail_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Desvio-padrão e amplitude (max-min) de offset_min/max por eixo, para um prefixo de colunas."""
    rows = []
    for axis in AXES:
        for side in ("min", "max"):
            values = detail_df[f"{prefix}_offset_{side}_{axis}"]
            rows.append(
                {
                    "eixo": axis,
                    "lado": side,
                    "std": float(values.std()),
                    "amplitude": float(values.max() - values.min()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-per-block",
        type=int,
        default=10,
        help="Casos por bloco do dataset (padrão 10 x 5 blocos = 50 no total)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed da amostragem estratificada")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignora candidate_anchor_z.csv já salvo e reprocessa tudo (padrão: retoma, pulando casos 'ok')",
    )
    args = parser.parse_args()

    DIAGNOSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config_json(str(CONFIG_PATH), {})

    all_cases = pd.read_csv(PRODUCTION_ALL_CASES)
    sample = stratified_sample(all_cases, args.sample_per_block, args.seed)
    print(
        f"Amostra: {len(sample)} casos "
        f"({args.sample_per_block} por bloco x {sample['block'].nunique()} blocos, seed={args.seed})"
    )

    candidate_path = DIAGNOSIS_OUTPUT_DIR / "candidate_anchor_z.csv"
    candidate_rows: list[dict] = []
    done_ids: set[int] = set()
    if candidate_path.exists() and not args.fresh:
        previous = pd.read_csv(candidate_path)
        candidate_rows = previous.to_dict("records")
        done_ids = set(previous.loc[previous["status"] == "ok", "img_id"])
        print(f"Retomando: {len(done_ids)} caso(s) já 'ok' em {candidate_path.name} serão pulados.")

    # Reprocessa sempre os que faltam ou que tinham dado erro antes (mesmo padrão retomável do
    # resto do projeto: só pula quem já terminou com sucesso).
    pending = sample[~sample["img_id"].isin(done_ids)]
    print(f"A processar nesta execução: {len(pending)} caso(s).\n")

    for i, row in enumerate(pending.itertuples(), 1):
        print(f"[{i}/{len(pending)}] IMG_ID={row.img_id} (bloco {row.block})...", end=" ", flush=True)
        started = time.perf_counter()
        result = compute_candidate_anchor_z(row.img_id, row.block, config)
        result["processing_seconds"] = time.perf_counter() - started
        # Substitui uma tentativa anterior com erro para o mesmo img_id, se houver.
        candidate_rows = [r for r in candidate_rows if r["img_id"] != row.img_id]
        candidate_rows.append(result)
        if result["status"] == "ok":
            print(
                f"z_min={result['z_min_candidate']:.1f} z_max={result['z_max_mask']:.1f} "
                f"({result['processing_seconds']:.1f}s)"
            )
        else:
            print(f"ERRO: {result['error_message']}")
        # Salva incrementalmente (custo desprezível para dezenas de linhas) para não perder
        # progresso se a execução for interrompida.
        pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df = candidate_df[candidate_df["img_id"].isin(sample["img_id"])]
    print(f"\nSalvo: {candidate_path}")

    # Checagem de sanidade: nenhuma inversão de lado (z_min sempre < z_max) entre os casos.
    ok_candidates = candidate_df[candidate_df["status"] == "ok"].copy()
    inverted = ok_candidates[ok_candidates["z_min_candidate"] >= ok_candidates["z_max_mask"]]
    if len(inverted) > 0:
        print(
            f"AVISO: {len(inverted)} caso(s) com z_min >= z_max (possível inversão de lado): "
            f"{inverted['img_id'].tolist()}"
        )
    else:
        print(f"Checagem de sanidade OK: z_min < z_max em todos os {len(ok_candidates)} casos.")

    merge_cols = [
        "img_id",
        "block",
        "z_min_candidate",
        "z_max_mask",
        "num_circles_detected",
        "processing_seconds",
    ]
    merged = sample.merge(ok_candidates[merge_cols], on=["img_id", "block"], how="inner")
    print(f"\nCasos com âncora candidata calculada com sucesso: {len(merged)} de {len(sample)}")

    detail_rows = []
    for row in merged.to_dict("records"):
        current_anchor = {"y": row["anchor_y"], "x": row["anchor_x"], "z": row["anchor_z"]}
        candidate_anchor = {"y": row["anchor_y"], "x": row["anchor_x"], "z": row["z_min_candidate"]}

        detail = {"img_id": row["img_id"], "block": row["block"]}
        detail.update(compute_offsets(row, current_anchor, "current"))
        detail.update(compute_offsets(row, candidate_anchor, "candidate"))
        detail_rows.append(detail)

    detail_df = pd.DataFrame(detail_rows)
    detail_path = DIAGNOSIS_OUTPUT_DIR / "offsets_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Salvo: {detail_path}")

    current_summary = summarize_dispersion(detail_df, "current").rename(
        columns={"std": "std_atual", "amplitude": "amplitude_atual"}
    )
    candidate_summary = summarize_dispersion(detail_df, "candidate").rename(
        columns={"std": "std_candidata", "amplitude": "amplitude_candidata"}
    )
    comparison = current_summary.merge(candidate_summary, on=["eixo", "lado"])
    comparison["reducao_std_%"] = (
        1 - comparison["std_candidata"] / comparison["std_atual"]
    ) * 100
    comparison["reducao_amplitude_%"] = (
        1 - comparison["amplitude_candidata"] / comparison["amplitude_atual"]
    ) * 100

    comparison_path = DIAGNOSIS_OUTPUT_DIR / "dispersion_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n=== Comparação de dispersão: âncora atual (centróide) vs. candidata (z_min) ===")
    print(comparison.round(2).to_string(index=False))
    print(f"\nSalvo: {comparison_path}")


if __name__ == "__main__":
    main()
