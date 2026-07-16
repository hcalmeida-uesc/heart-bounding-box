"""Execução em lote retomável, genérica sobre qualquer função de processamento por caso.

Extraído de ``segmentation.envelope`` para ser reaproveitado por outras etapas do bounding box
cardíaco (ex.: calibração de âncora via aorta) sem duplicar a lógica de checkpoint incremental.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

ProcessCaseFn = Callable[[int, "str | Path", str, dict[str, Any]], dict[str, Any]]


def run_resumable_batch(
    process_case_fn: ProcessCaseFn,
    record_columns: list[str],
    block: str,
    img_ids: list[int],
    base_path: str | Path,
    config: dict[str, Any],
    output_csv: str | Path,
    resume: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Processa uma lista de casos, salvando incrementalmente em ``output_csv``.

    ``process_case_fn(img_id, base_path, block, config) -> dict`` é chamada por caso; o dict
    retornado deve conter ao menos as chaves ``img_id`` e ``status`` (``"ok"``/``"error"``) para
    que a retomada funcione. Casos já registrados com ``status="ok"`` em uma execução anterior são
    pulados quando ``resume=True`` (default); casos com erro são sempre re-tentados.
    """
    output_csv = Path(output_csv)

    records_by_id: dict[int, dict[str, Any]] = {}
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        records_by_id = {int(row["img_id"]): row.to_dict() for _, row in existing.iterrows()}

    done_ids = {
        img_id for img_id, record in records_by_id.items() if record.get("status") == "ok"
    }
    pending_ids = [img_id for img_id in img_ids if img_id not in done_ids]

    iterator = pending_ids
    if progress:
        from tqdm import tqdm

        iterator = tqdm(pending_ids, desc=f"bloco {block}")

    for img_id in iterator:
        record = process_case_fn(img_id, base_path, block, config)
        records_by_id[img_id] = record

        df = pd.DataFrame(records_by_id.values()).reindex(columns=record_columns)
        df = df.sort_values("img_id").reset_index(drop=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    result = pd.DataFrame(records_by_id.values()).reindex(columns=record_columns)
    return result.sort_values("img_id").reset_index(drop=True)


__all__ = ["run_resumable_batch"]
