"""Executa a etapa 6 (envelope de dimensões máximas) sobre o dataset local completo.

Uso:
    uv run python src/run_envelope.py            # roda/retoma o envelope sobre todo o dataset
    uv run python src/run_envelope.py --fresh     # ignora progresso salvo e reprocessa tudo

Processa cada bloco (subpasta) encontrado em ``dataset/`` — na convenção usada pelo ImageCAS local
(``1-200``, ``201-400``, ...) — salvando um CSV por bloco em ``output/bounding_box/envelope/``.
O diretório de saída é fixo (não tem timestamp) para que reexecutar o script retome o progresso em
vez de recomeçar do zero; ver ``process_block`` em ``utils/segmentation/envelope.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.project.config import load_config_json
from utils.segmentation.envelope import compute_envelope, run_envelope_batch

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
    output_dir = REPO_ROOT / "output" / "bounding_box" / "envelope"
    print(f"Saída: {output_dir}")

    records_df = run_envelope_batch(
        dataset_root, config, output_dir, resume=not args.fresh, progress=True
    )
    records_df.to_csv(output_dir / "all_cases.csv", index=False)

    num_errors = int((records_df["status"] == "error").sum())
    if num_errors:
        print(f"AVISO: {num_errors} caso(s) com erro — ver coluna error_message em all_cases.csv")

    envelope_summary = compute_envelope(records_df, config["DOWNSCALE_FACTORS"])
    with (output_dir / "envelope_summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(envelope_summary, file_handle, indent=2, ensure_ascii=False)

    print(json.dumps(envelope_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
