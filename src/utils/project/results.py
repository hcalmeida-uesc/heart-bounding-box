"""Utilities genéricos de timing e persistência de execuções em lote."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def make_json_safe(value: Any) -> Any:
    """Converte valores comuns de pandas/numpy/pathlib para JSON nativo."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "as_posix"):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def duration_breakdown(duration_seconds: Any) -> dict[str, Any]:
    """Retorna duração em segundos, minutos e horas."""
    if duration_seconds is None or pd.isna(duration_seconds):
        return {
            "seconds": None,
            "minutes": None,
            "hours": None,
        }

    seconds = float(duration_seconds)
    return {
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
    }


def create_timestamped_output_dir(base_output_dir, experiment_name="run"):
    """Cria diretório de saída com timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(base_output_dir) / experiment_name / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


BATCH_TIMING_COLUMNS = [
    "split_name",
    "batch_number",
    "total_batches",
    "num_images",
    "first_img_id",
    "last_img_id",
    "result_file",
    "started_at",
    "finished_at",
    "duration_seconds",
    "duration_minutes",
    "duration_hours",
]


def batch_timing_manifest_path(output_dir, split_name) -> Path:
    """Retorna o caminho do CSV com tempos por lote."""
    return Path(output_dir) / f"{split_name}_batch_timings.csv"


def load_batch_timing_records(output_dir, split_name) -> list[dict[str, Any]]:
    """Carrega tempos por lote já salvos."""
    manifest_path = batch_timing_manifest_path(output_dir, split_name)
    if not manifest_path.exists():
        return []

    df = pd.read_csv(manifest_path)
    if df.empty:
        return []
    return df.to_dict("records")


def save_batch_timing_record(
    output_dir,
    split_name,
    record: dict[str, Any],
) -> str:
    """Insere ou atualiza o tempo de um lote no manifest incremental."""
    manifest_path = batch_timing_manifest_path(output_dir, split_name)
    records = load_batch_timing_records(output_dir, split_name)
    batch_number = int(record["batch_number"])

    records = [
        item
        for item in records
        if int(item.get("batch_number", -1)) != batch_number
    ]
    records.append(record)
    records = sorted(records, key=lambda item: int(item["batch_number"]))

    df = pd.DataFrame(records)
    df = df.reindex(columns=BATCH_TIMING_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return str(manifest_path)


def summarize_batch_timing_records(
    records: list[dict[str, Any]],
    expected_batches: list[int] | None = None,
) -> dict[str, Any]:
    """Soma tempos conhecidos dos lotes e marca lotes sem tempo salvo."""
    if not records:
        return {
            "total_known_batches": 0,
            "total_known_duration_seconds": None,
            "total_known_duration_minutes": None,
            "total_known_duration_hours": None,
            "missing_timing_batches": expected_batches or [],
        }

    durations_by_batch = {}
    for record in records:
        batch_number = int(record["batch_number"])
        duration = pd.to_numeric(record.get("duration_seconds"), errors="coerce")
        if not pd.isna(duration):
            durations_by_batch[batch_number] = float(duration)

    known_batches = sorted(durations_by_batch)
    total_seconds = sum(durations_by_batch.values()) if durations_by_batch else None

    missing_timing_batches = []
    if expected_batches is not None:
        missing_timing_batches = [
            batch_number
            for batch_number in expected_batches
            if batch_number not in durations_by_batch
        ]

    total = duration_breakdown(total_seconds)
    return {
        "total_known_batches": len(known_batches),
        "known_batches": known_batches,
        "total_known_duration_seconds": total["seconds"],
        "total_known_duration_minutes": total["minutes"],
        "total_known_duration_hours": total["hours"],
        "missing_timing_batches": missing_timing_batches,
    }
