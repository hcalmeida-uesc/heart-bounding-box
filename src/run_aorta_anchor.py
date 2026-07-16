"""Executa a etapa 6 v2 (âncora pela aorta) sobre o dataset local completo.

Uso:
    uv run python src/run_aorta_anchor.py            # roda/retoma sobre todo o dataset
    uv run python src/run_aorta_anchor.py --fresh     # ignora progresso salvo e reprocessa tudo

Processa cada bloco (subpasta) de ``dataset/``, salvando um CSV por bloco em
``output/bounding_box/aorta_anchor/`` (retomável — reexecutar pula os casos já processados com
sucesso). Em seguida calibra os offsets fixos do bbox usando só o split de treino
(``config/imagecas_splits.json``) e valida a cobertura da árvore coronária nos splits de
validação/teste (sem vazamento entre calibração e validação).

Aviso de custo: cada caso roda detecção de círculos (Hough) + Level Set da aorta — ~35s/caso nesta
máquina sem GPU, ou seja, ~9-10h para as 1000 imagens. Os módulos já suportam GPU
(``"USE_GPU": true`` em ``config/pipeline_config.json``, requer cupy/cucim) — considerar rodar na
máquina com GPU compartilhada do grupo em vez desta, ou deixar rodando em background aqui.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.project.config import load_config_json
from utils.segmentation.aorta_anchor import (
    calibrate_bbox_offsets,
    run_aorta_anchor_batch,
    validate_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignora CSVs de progresso já salvos e reprocessa todos os casos.",
    )
    args = parser.parse_args()

    dataset_root = REPO_ROOT / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {dataset_root}. Confirme o symlink/junction 'dataset' "
            "na raiz do repositório (ver .gitignore)."
        )

    config = load_config_json(str(REPO_ROOT / "config" / "pipeline_config.json"), {})
    output_dir = REPO_ROOT / "output" / "bounding_box" / "aorta_anchor"
    print(f"Saída: {output_dir}")

    records_df = run_aorta_anchor_batch(
        dataset_root, config, output_dir, resume=not args.fresh, progress=True
    )
    records_df.to_csv(output_dir / "all_cases.csv", index=False)

    num_errors = int((records_df["status"] == "error").sum())
    if num_errors:
        print(
            f"AVISO: {num_errors} caso(s) sem âncora detectada — ver error_message em "
            "all_cases.csv (excluídos da calibração/validação abaixo)."
        )

    # Lê o split fixo diretamente (em vez de get_data_splits, que espera um único diretório plano
    # com as 1000 imagens — aqui o dataset está em blocos por pasta, ver discover_dataset_blocks).
    splits_path = REPO_ROOT / "config" / "imagecas_splits.json"
    splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
    train_ids = [int(img_id) for img_id in splits_data["splits"]["train"]]
    val_ids = [int(img_id) for img_id in splits_data["splits"]["val"]]
    test_ids = [int(img_id) for img_id in splits_data["splits"]["test"]]

    all_ids = set(records_df["img_id"])
    missing_from_dataset = (set(train_ids) | set(val_ids) | set(test_ids)) - all_ids
    if missing_from_dataset:
        raise ValueError(
            f"IDs do split fixo ausentes no dataset processado: {sorted(missing_from_dataset)[:10]}"
        )

    train_records = records_df[records_df["img_id"].isin(train_ids)]
    calibration = calibrate_bbox_offsets(train_records, config["DOWNSCALE_FACTORS"])
    with (output_dir / "calibration.json").open("w", encoding="utf-8") as file_handle:
        json.dump(calibration, file_handle, indent=2, ensure_ascii=False)
    print("Calibração (split de treino):")
    print(json.dumps(calibration, indent=2, ensure_ascii=False))

    for split_name, split_ids in (("val", val_ids), ("test", test_ids)):
        split_records = records_df[records_df["img_id"].isin(split_ids)]
        coverage_df = validate_coverage(split_records, calibration)
        coverage_df.to_csv(output_dir / f"coverage_{split_name}.csv", index=False)
        coverage_rate = coverage_df["covered"].mean() if len(coverage_df) else float("nan")
        print(
            f"Cobertura no split '{split_name}': {coverage_df['covered'].sum()}/"
            f"{len(coverage_df)} casos ({coverage_rate:.2%})"
        )


if __name__ == "__main__":
    main()
