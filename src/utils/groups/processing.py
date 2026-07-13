"""Grouped exports for image pre-processing utilities - lazy loaded."""

from importlib import import_module

_SYMBOL_TO_MODULE = {
    # preprocessing
    "downscale_image_ndi": "preprocessing",
    "downscale_image_opencv": "preprocessing",
    "largest_connected_component": "preprocessing",
    "run_core_preprocessing_pipeline": "preprocessing",
    "threshold_image": "preprocessing",
    "threshold_image_with_offset": "preprocessing",
}


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        submodule = _SYMBOL_TO_MODULE[name]
        parent_pkg = __package__.rpartition(".")[0] or __package__
        module = import_module(f".processing.{submodule}", parent_pkg)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
