"""Subpacote de segmentação com exports públicos carregados sob demanda."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # lower_threshold
    "get_lower_threshold_config": "lower_threshold",
    "normalize_lower_threshold_method": "lower_threshold",
    "resolve_lower_threshold": "lower_threshold",
    # pipeline_preprocessing
    "load_and_preprocess_image": "pipeline_preprocessing",
    # bounding_box
    "compute_geometric_centroid": "bounding_box",
    "compute_bounding_extent": "bounding_box",
    # envelope
    "discover_dataset_blocks": "envelope",
    "process_case_for_envelope": "envelope",
    "process_block": "envelope",
    "run_envelope_batch": "envelope",
    "compute_envelope": "envelope",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(f".{_SYMBOL_TO_MODULE[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
