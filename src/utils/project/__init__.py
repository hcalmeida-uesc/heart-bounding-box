"""Utilitários de infraestrutura do projeto.

Este subpacote agrupa funções que não pertencem diretamente a processamento,
segmentação ou visualização: cache em disco, configuração, splits do dataset,
ambiente de notebooks e persistência de resultados.
"""

from .cache import load_json_cache, load_npy_cache, save_json_cache, save_npy_cache
from .config import (
    deep_update_dict,
    load_config_json,
    normalize_runtime_config,
    save_config_json,
    serialize_config_for_json,
)
from .dataset import get_data_splits
from .notebook_env import (
    configure_notebook_environment,
    resolve_existing_path,
    resolve_imagecas_base_path,
    resolve_processed_imagecas_path,
)
from .results import (
    create_timestamped_output_dir,
    duration_breakdown,
    load_batch_timing_records,
    make_json_safe,
    save_batch_timing_record,
    summarize_batch_timing_records,
)

__all__ = [
    "configure_notebook_environment",
    "create_timestamped_output_dir",
    "deep_update_dict",
    "duration_breakdown",
    "get_data_splits",
    "load_batch_timing_records",
    "load_config_json",
    "load_json_cache",
    "load_npy_cache",
    "make_json_safe",
    "normalize_runtime_config",
    "resolve_existing_path",
    "resolve_imagecas_base_path",
    "resolve_processed_imagecas_path",
    "save_batch_timing_record",
    "save_config_json",
    "save_json_cache",
    "save_npy_cache",
    "serialize_config_for_json",
    "summarize_batch_timing_records",
]
