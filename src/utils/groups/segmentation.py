"""Grouped exports for segmentation-related utilities - lazy loaded."""

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
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        parent_pkg = __package__.rpartition(".")[0] or __package__
        module = import_module(f".segmentation.{submodule}", parent_pkg)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
